import java.util.Arrays;

public class LetterHistogram {
    public static void main(String[] args) {
        System.out.println(Arrays.toString(getHistogram("6s84f6s68seannanananannnannana")));
    }

    public static int[] getHistogram(String word) {
        int[] histogram = new int[27];

        for (char c : word.toCharArray()) {
            int cIndex = letterIndex(c);
            histogram[cIndex]++;
        }

        return histogram;
    }

    public static int letterIndex(char inputChar) {
        // switch (inputChar) {
        //     case 'a':
        //         return 0;
        //     case 'b':
        //         return 1;
        //     ...
        //     case 'z':
        //         return 25;
        //     default:
        //         return 26;
        // }

        char[] alphabet = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'};
        for (int index = 0; index < alphabet.length; index++) {
            if (alphabet[index] == inputChar) {
                return index;
            }
        }
        return 26;
    }
}
