public class EarningsCalculator2 {

    public static double earnings(double hours, double wage, double factor) {
        // return hours <= 8.0
        //     ? hours * wage
        //     : (8.0 + factor * (hours - 8.0)) * wage;

        if (hours <= 8.0) {
            return hours * wage;
        } else {
            return (8.0 + factor * (hours - 8.0)) * wage;
        }
    }


    public static void main(String[] args) {
        final double wage = 15.0;
        final double factor = 1.15;

        // double hoursMon = 8.0;
        // double hoursTue = 8.0;
        // double hoursWed = 9.0;
        // double hoursThur = 9.0;
        // double hoursFri = 6.0;
        // double hoursSat = 8.0;

        double[] timeSpent = {8, 8, 9, 9, 6, 8};


        double total = 0;
        for (double workingHours : timeSpent) {
            double moneyEarned = earnings(workingHours, wage, factor);
            total += moneyEarned;
        }
        System.out.println(total);


        // double total =
        //     earnings(hoursMon, wage, factor) +
        //     earnings(hoursTue, wage, factor) +
        //     earnings(hoursWed, wage, factor) +
        //     earnings(hoursThur, wage, factor) +
        //     earnings(hoursFri, wage, factor) +
        //     earnings(hoursSat, wage, factor);

    }
}