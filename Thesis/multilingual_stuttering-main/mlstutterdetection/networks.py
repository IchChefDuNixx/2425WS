from typing import Optional, Union, Tuple
from fvcore.nn.focal_loss import sigmoid_focal_loss
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Config, Wav2Vec2Model
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import _HIDDEN_STATES_START_POSITION, Wav2Vec2BaseModelOutput


class AttentionCLSHead(nn.Module):
    def __init__(self, emb_dim, num_labels, att_dim=768, att_heads=1):
        super().__init__()

        self.seq_query = torch.nn.init.xavier_normal_(torch.nn.Parameter(torch.zeros(1, att_dim)))
        self.attention = nn.MultiheadAttention(embed_dim=att_dim, num_heads=att_heads, batch_first=True, dropout=0.5)
        self.proj = nn.Linear(att_dim, emb_dim)
        self.clf = nn.Linear(emb_dim, num_labels)

    def forward(self, x):
        """x is a sequence of wav2vec 2.0 embeddings, batch dim is dim 0 """
        # pool, _ = self.attention(self.seq_query, x, x)
        pool, _ = self.attention(torch.stack([self.seq_query for x in range(x.shape[0])]), x, x)
        pool = pool.squeeze(1) # remove seq dim
        pool = self.proj(pool)
        pool = F.relu(pool)
        # pool = self.layer_norm(pool)
        return self.clf(pool)


class Wav2Vec2ForSequenceClassificationAuxAttentionHead(Wav2Vec2ForSequenceClassification):
    """ don't use this impl, better use the CLS head version, outperforms """
    def __init__(self, config, num_aux_labels=2, stat_pooling=False, class_weights=None, loss=None, att_heads=1):

        super().__init__(config)
        self.stat_pooling = stat_pooling
        linear_size = 2 * config.classifier_proj_size if stat_pooling else config.classifier_proj_size

        self.seq_query = torch.nn.init.xavier_normal_(torch.nn.Parameter(torch.zeros(1, linear_size)))
        self.attention = nn.MultiheadAttention(embed_dim=linear_size, num_heads=att_heads, batch_first=True, dropout=0.5) #TODO: seems a bit high

        self.classifier = nn.Linear(linear_size, config.num_labels)
        self.aux = nn.Linear(linear_size, num_aux_labels)

        self.num_aux_labels = num_aux_labels
        self.class_weights = class_weights
        # can be trained with simple CEL, just don't set loss
        self.loss = CrossEntropyLoss(weight=self.class_weights) if loss is None else loss

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
            aux_labels=None
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        hidden_states = self.projector(hidden_states)

        pooled_output, _ = self.attention(torch.stack([self.seq_query for _ in range(hidden_states.shape[0])]),
                                          hidden_states, hidden_states)

        logits = self.classifier(pooled_output.squeeze(1))  # Squeeze obsolete seq dimension
        aux_logits = self.aux(pooled_output.squeeze(1))
        clf_loss, aux_loss, loss = None, None, None

        if labels is not None and aux_labels is None:
            loss = self.loss(logits, labels)
        if labels is not None and aux_labels is not None:
            loss = self.loss(logits=logits, aux_logits=aux_logits, labels=labels, aux_labels=aux_labels)

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# from pretrained model erben und selber classification head?
class Wav2Vec2ForSequenceClassificationAux(Wav2Vec2ForSequenceClassification):
    def __init__(self, config, num_aux_labels=2, stat_pooling=False, class_weights=None, loss=None):

        super().__init__(config)
        self.stat_pooling = stat_pooling
        linear_size = 2 * config.classifier_proj_size if stat_pooling else config.classifier_proj_size
        self.classifier = nn.Linear(linear_size, config.num_labels)
        self.aux = nn.Linear(linear_size, num_aux_labels)
        self.num_aux_labels = num_aux_labels
        self.class_weights = class_weights
        # can be trained with simple CEL, just don't set loss
        self.loss = CrossEntropyLoss(weight=self.class_weights) if loss is None else loss

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
            aux_labels=None
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        hidden_states = self.projector(hidden_states)

        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1) if not self.stat_pooling else torch.cat(
                torch.std_mean(hidden_states, dim=1), dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        logits = self.classifier(pooled_output)
        aux_logits = self.aux(pooled_output)
        clf_loss, aux_loss, loss = None, None, None

        if labels is not None and aux_labels is None:
            loss = self.loss(logits, labels)
        if labels is not None and aux_labels is not None:
            loss = self.loss(logits=logits, aux_logits=aux_logits, labels=labels, aux_labels=aux_labels)

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Wav2Vec2ForSequenceClassificationAuxDeep(Wav2Vec2ForSequenceClassification):
    def __init__(self, config, num_aux_labels=2, stat_pooling=False, class_weights=None, loss=None):

        super().__init__(config)
        self.stat_pooling = stat_pooling
        linear_size = 2 * config.hidden_size if stat_pooling else config.hidden_size

        self.projector = nn.Linear(linear_size, config.classifier_proj_size)
        self.hidden = nn.Linear(config.classifier_proj_size, config.classifier_proj_size // 2)
        self.classifier = nn.Linear(config.classifier_proj_size // 2, config.num_labels)

        self.aux = nn.Linear(config.classifier_proj_size // 2, num_aux_labels)

        self.num_aux_labels = num_aux_labels
        self.class_weights = class_weights
        # can be trained with simple CEL, just don't set loss
        self.loss = CrossEntropyLoss(weight=self.class_weights) if loss is None else loss

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
            aux_labels=None
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1) if not self.stat_pooling else torch.cat(
                torch.std_mean(hidden_states, dim=1), dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        pooled_output = self.projector(F.relu(pooled_output))
        pooled_output = self.hidden(F.relu(pooled_output))
        logits = self.classifier(pooled_output)

        aux_logits = self.aux(pooled_output)
        clf_loss, aux_loss, loss = None, None, None

        if labels is not None and aux_labels is None:
            loss = self.loss(logits, labels)
        if labels is not None and aux_labels is not None:
            loss = self.loss(logits=logits, aux_logits=aux_logits, labels=labels, aux_labels=aux_labels)

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MTLCrossEntropy(nn.Module):
    def __init__(self, main_loss_weight, class_weights, aux_class_weights, num_labels, num_aux_labels):
        super().__init__()
        self.main_loss_weight = main_loss_weight
        self.clf_loss_fct = CrossEntropyLoss(weight=class_weights)
        self.aux_loss_fct = CrossEntropyLoss(weight=aux_class_weights)
        self.num_labels = num_labels
        self.num_aux_labels = num_aux_labels

    def forward(self, logits, aux_logits, labels, aux_labels):
        clf_loss = self.clf_loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        aux_loss = self.aux_loss_fct(aux_logits.view(-1, self.num_aux_labels), aux_labels.view(-1))
        loss = clf_loss * self.main_loss_weight + aux_loss * (1 - self.main_loss_weight)
        return loss


class MultiClassMTLCrossEntropy(nn.Module):
    def __init__(self, main_loss_weight, class_weights, pos_weight, aux_class_weights, num_labels, num_aux_labels):
        super().__init__()
        self.main_loss_weight = main_loss_weight
        self.clf_loss_fct = BCEWithLogitsLoss(weight=class_weights, reduction='mean', pos_weight=pos_weight)
        self.aux_loss_fct = CrossEntropyLoss(weight=aux_class_weights)
        self.num_labels = num_labels
        self.num_aux_labels = num_aux_labels

    def forward(self, logits, aux_logits, labels, aux_labels):
        clf_loss = self.clf_loss_fct(logits, labels)
        aux_loss = self.aux_loss_fct(aux_logits.view(-1, self.num_aux_labels), aux_labels.view(-1))
        loss = clf_loss * self.main_loss_weight + aux_loss * (1 - self.main_loss_weight)
        return loss


class MultiClassMTLFocalLoss(nn.Module):
    def __init__(self, main_loss_weight, alpha, gamma, aux_class_weights, num_aux_labels, reduction='mean'):
        super().__init__()
        self.main_loss_weight = main_loss_weight
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.aux_loss_fct = CrossEntropyLoss(weight=aux_class_weights)
        self.num_aux_labels = num_aux_labels

    def forward(self, logits, aux_logits, labels, aux_labels):
        clf_loss = sigmoid_focal_loss(logits, labels, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
        aux_loss = self.aux_loss_fct(aux_logits.view(-1, self.num_aux_labels), aux_labels.view(-1))
        loss = clf_loss * self.main_loss_weight + aux_loss * (1 - self.main_loss_weight)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        inputs = logits.float()
        targets = labels.float()
        p = torch.sigmoid(inputs)
        # don't add pos weights here, just leads to recall 1 and precision 0
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class Wav2Vec2CLSpooling(Wav2Vec2Model):
    def __init__(self, config: Wav2Vec2Config, pooling_mechanism=None):
        super().__init__(config)
        self.pooling = get_cls_pooling_mechanism(pooling_mechanism)

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)

        extract_features = extract_features.transpose(1, 2)
        extract_features = self.pooling(extract_features)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class Wav2Vec2ForSequenceClassificationCLSAux(Wav2Vec2ForSequenceClassification):
    def __init__(self, config, num_aux_labels=2, pooling_mechanism='projection', class_weights=None, loss=None,
                 clf_dropout=False):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2CLSpooling(config, pooling_mechanism=pooling_mechanism)
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)
        self.aux = nn.Linear(config.classifier_proj_size, num_aux_labels)
        self.num_aux_labels = num_aux_labels
        self.dropout = nn.Dropout(0.2) if clf_dropout else nn.Identity()
        self.class_weights = class_weights
        # can be trained with simple CEL, just don't set loss
        self.loss = CrossEntropyLoss(weight=self.class_weights) if loss is None else loss

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
            aux_labels=None
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        # the pooling mech returns the 0th element as the pooled one, projection only on this
        pooled_output = self.projector(hidden_states[:, 0, :])  # use only first token for classification in CLS pooling
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        aux_logits = self.aux(pooled_output)
        clf_loss, aux_loss, loss = None, None, None

        if labels is not None and aux_labels is None:
            loss = self.loss(logits, labels)
        if labels is not None and aux_labels is not None:
            loss = self.loss(logits=logits, aux_logits=aux_logits, labels=labels, aux_labels=aux_labels)

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class W2v2CLSMeanPooling(nn.Module):
    # probably nonsense, nothing is learnt here, only in later layers, as there is one vector more, but no init projection
    def __init__(self, emb_dim=768):
        # emb dim for coherence with api
        super().__init__()

    def forward(self, x):
        return torch.cat([torch.mean(x, dim=1, keepdim=True), x], dim=1)


class W2v2CLSMeanProjectionPooling(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        pool = self.proj(x)
        pool = self.gelu(pool)
        pool = self.layer_norm(pool)
        return torch.cat([pool, torch.mean(x, dim=1, keepdim=True)], dim=1)  # append pooled input to the front of sequence


class W2v2CLSAttentionPooling(nn.Module):
    def __init__(self, emb_dim=512, att_heads=1):
        super().__init__()
        self.gelu = nn.GELU()
        self.attention = nn.MultiheadAttention(emb_dim, att_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        # pool w.r.t. the mean vector of the sequence
        pool, _ = self.attention(torch.mean(x, dim=1, keepdim=True), x, x)
        pool = self.layer_norm(pool)
        pool = self.gelu(pool)
        return torch.cat([pool, x], dim=1)  # append pooled input to the front of sequence


class W2v2CLSTokenPooling(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()
        self.proj = torch.nn.Parameter(torch.zeros(1, emb_dim))
        _ = torch.nn.init.xavier_normal_(self.proj)

    def forward(self, x):
        """ x must be batch first for this to work"""
        seq_tokens = torch.stack([self.proj for _ in range(x.shape[0])])
        # append seq token to front of sequence
        return torch.cat([seq_tokens, x], dim=1)


cls_pooling_mechs = {
    # 'mean': W2v2CLSMeanPooling,
    'token': W2v2CLSTokenPooling(),
    'projection': W2v2CLSMeanProjectionPooling(),
    'attention': W2v2CLSAttentionPooling(),
}


def get_cls_pooling_mechanism(pooling):
    return cls_pooling_mechs.get(pooling)
