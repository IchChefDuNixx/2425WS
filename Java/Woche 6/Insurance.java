public class Insurance {
    static float policy1(float total) { return 50 + total * 0.04f; }
    static float policy2(float total) { return total * 0.05f; }
    static float policy3(float total) { return 100 + total * 0.035f; }

    public static void main(String[] args) {

        float ins1, ins2, ins3, best, diff;
        String bestName;
        for (int price = 1000; price < 12999; price += 200) {
            ins1 = policy1(price);
            ins2 = policy2(price);
            ins3 = policy3(price);

            // initial assumption
            best = ins1;
            bestName = "Ins.1";
            diff = 0;

            // compare to second insurance
            if (ins2 == best) {
                bestName += "+Ins.2";
                diff = 0;
            } else if (ins2 < best) {
                diff = best - ins2;
                best = ins2;
                bestName = "Ins.2";
            } else {
                if (ins2 < ins3) {
                    diff = ins2 - best;
                } else {
                    diff = ins3 - best;
                }
            }

            // compare to third insurance
            if (ins3 == best) {
                bestName += "+Ins.3";
                diff = 0;
            } else if (ins3 < best) {
                diff = best - ins3;
                best = ins3;
                bestName = "Ins.3";
            }

            // results
            System.out.printf(
                "%d\t%2.0f\t%2.0f\t%2.0f\t%s\t%2.0f%n",
                price, ins1, ins2, ins3, bestName, diff
            );
        }
    }
}