import java.util.Arrays;

public class Append {
    public static void main(String[] args) {
        int[] abc = {684,684,684,6,5,4,84,3,54,68,4};
        System.out.println(Arrays.toString(append(abc, 50)));
    }

    public static int[] append(int[] arr, int extra) {
        int[] result = new int[arr.length + 1];
        for (int i = 0; i < arr.length; i++) {
            result[i] = arr[i];
        }
        result[arr.length] = extra;
        return result;
    }
}
