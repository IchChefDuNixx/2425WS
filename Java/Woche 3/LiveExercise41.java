public class LiveExercise41 {

    public static void sop(String s) {
        System.out.println(s);
        return;
    }

    public static void sop(double d) {
        System.out.println(d);
        return;
    }

    public static void sop(int i) {
        System.out.println(i);
        return;
    }

    public static double compoundInterest(double startingBalance, double basicInterestRate, int investmentYears) {
        double capital = startingBalance * Math.pow(1 + basicInterestRate/100, investmentYears);
        return capital;
    }

    public static double compoundInterestWithInterestRise(double startingBalance, double basicInterestRate, int investmentYears, double rise) {
        double capital = compoundInterest(startingBalance, basicInterestRate, investmentYears);
        return capital * (1 + rise);
    }

    public static void main(String[] args) {

        // String a = "Hallo";
        // sop(a);
        // sop("Welt");
        //
        // System.out.print("Hallo Welt\n");
        // System.out.println("Hallo Welt");

        double myMoney = 5555;
        double interest = 0.04;
        int time = 10;
        double rise = 0.01;

        sop(compoundInterest(myMoney, interest, time));
        sop(compoundInterestWithInterestRise(myMoney, interest, time, rise));

    }


}
