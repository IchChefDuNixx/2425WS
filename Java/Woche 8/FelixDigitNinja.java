import java.util.Arrays;

public class FelixDigitNinja {
    public static int[] histogramDings(String s) {
        int[] hist = new int[10];

        for (char c : s.toCharArray()) {
            if (Character.isDigit(c)) {
                hist[Character.getNumericValue(c)]++;
            }
        }

        return hist;
    }

    public static void main(String[] args) {
        System.out.print(Arrays.toString(histogramDings("123456789")));

    }
}