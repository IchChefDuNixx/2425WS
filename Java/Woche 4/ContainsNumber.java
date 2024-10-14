public class ContainsNumber {

    public static void main(String[] args) {

    }

    public static boolean containsNumber(double[] data, double number) {

        for (int counter = 0; counter < data.length; counter++) {
            if (data[counter] == number) {
                return true;
            }
        }

        return false;

    }
}
