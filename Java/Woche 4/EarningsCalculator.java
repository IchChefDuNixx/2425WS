import java.util.Arrays;

public class EarningsCalculator {

    public static double earnings(double hours, double wage, double factor) {
        // return hours <= 8.0
        //     ? hours * wage
        //     : (8.0 + factor * (hours - 8.0)) * wage;

        // ternary if statement:
        // [condition] ? [execute if true] : [execute if false]

        if (hours <= 8.0) {
            return hours * wage;
        } else {
            return (8.0 + factor * (hours - 8.0)) * wage;
        }
    }

    public static double[] append(double[] arr, double extra) {
        double[] result = new double[arr.length + 1];
        for (int i = 0; i < arr.length; i++) {
            result[i] = arr[i];
        }
        result[arr.length] = extra;
        return result;
    }

    public static double[] remove(double[] arr, int index) {
        double[] result = new double[arr.length - 1];
        for (int i = 0; i < index; i++) {
            result[i] = arr[i];
        }
        for (int i = index + 1; i < arr.length; i++) {
            result[i - 1] = arr[i];
        }
        return result;
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
        timeSpent = append(timeSpent, 5);
        timeSpent = append(timeSpent, 5);
        System.out.println(Arrays.toString(timeSpent));
        timeSpent = remove(timeSpent, 6);
        timeSpent = remove(timeSpent, 0);
        System.out.println(Arrays.toString(timeSpent));


        // double total =
        //     earnings(hoursMon, wage, factor) +
        //     earnings(hoursTue, wage, factor) +
        //     earnings(hoursWed, wage, factor) +
        //     earnings(hoursThur, wage, factor) +
        //     earnings(hoursFri, wage, factor) +
        //     earnings(hoursSat, wage, factor);

        double total = 0;
        for (double time : timeSpent) {
            total += earnings(time, wage, factor);
        }

        System.out.println(total);
    }
}