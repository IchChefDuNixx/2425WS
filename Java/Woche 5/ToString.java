import java.util.Arrays;

public class ToString {
    public static void main(String[] args) {
        double[] testArr = new double[]{0,84,0.8646,0.6846,684.543};

        // original implementation
        System.out.println(Arrays.toString(testArr));

        // my version
        customToString(testArr);
    }


    public static void customToString(double[] arr) {
        String result = "";
        for (int i = 0; i < arr.length - 1; i++) {
            result += arr[i] + ", ";
        }
        result += arr[arr.length-1];
        result = "[" + result + "]";

        System.out.println(result);
    }
}
