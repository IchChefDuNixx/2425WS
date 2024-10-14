import java.util.Scanner;

public class Change {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Please enter the price: ");
        double input1 = Double.parseDouble(sc.nextLine());
        System.out.print("Please enter the amount: ");
        double input2 = Double.parseDouble(sc.nextLine());

        // (int) casting: converting a variable of type double to an integer (260.00 -> 260, 300.00 -> 300)
        int input1_int = (int)(input1 * 100);
        int input2_int = (int)(input2 * 100);

        boolean isEnough = change(input1_int, input2_int);
        if (isEnough) {
            System.out.println("Thank you very much");
        } else {
            System.out.println("Unfortunately that's not enough");
        }
    }


    public static boolean change(int price, int paid) {
        if (price > paid) {
            return false;
        }

        int[] bankNotes = new int[]{10000, 5000, 2000, 1000, 500, 200, 100, 50, 20, 10, 5, 2, 1};
        int remainder = paid - price;
        System.out.println("Change: ");

        // Scheine
        for (int i = 0; i < bankNotes.length; i++) {
            int numBankNotes = remainder / bankNotes[i];
            if (numBankNotes > 0) {
                remainder -= numBankNotes * bankNotes[i];
                if (i <= 6) { // Scheine + 1/2€
                    System.out.println(numBankNotes + "x " + bankNotes[i]/100 + "eur");
                } else { // Kleingeld
                    System.out.println(numBankNotes + "x " + bankNotes[i] + "ct");
                }
            }
        }

        return true;
    }
}
