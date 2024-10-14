import java.util.Scanner;

public class Maximum {

    public static int max(double[] arr) {
        int currentMaxIndex = 0;
        double currentMaxValue = arr[0];

        for (int i = 0; i < arr.length; i++) {
            if (arr[i] > currentMaxValue) {
                currentMaxValue = arr[i];
                currentMaxIndex = i;
            }
        }

        System.out.println("The maximum is at index: " + currentMaxIndex);
        System.out.println("The maximum is: " + currentMaxValue);
        return currentMaxIndex;
    }


    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.println("Please enter a number of random numbers:");
        int num = Integer.parseInt(sc.nextLine());

        double[] randomNumbers = new double[num];
        for (int i = 0; i < randomNumbers.length; i++) {
            double rng = Math.random();
            randomNumbers[i] = rng;
            System.out.println("Random Number = " + rng);
        }

        max(randomNumbers);
    }

}
