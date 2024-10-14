public class Round {
    public static void main(String[] args) {

        // https://docs.oracle.com/javase/6/docs/api/java/util/Formatter.html#syntax

        double value = 99.68946846;
        String custom_format = "File size: %8.2f MB\n";

        System.out.printf(custom_format, value);
        System.out.print(String.format(custom_format, value/10));
        System.out.print(custom_format.formatted(value*100));



        double[] ar = {1.2, 3.0, 0.8};
        double sum = 0.0;
        for (int i = 0; i < ar.length; i++) {
            sum += ar[i];
        }

        for (double currentArrayElement : ar) {
            sum += currentArrayElement;
        }
        System.out.println(sum);
    }
}
