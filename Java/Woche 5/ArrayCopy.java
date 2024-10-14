import java.util.Arrays;

public class ArrayCopy {
    public static void main(String[] args) {
        int[] arrA = new int[]{1,6,2,9,0};
        int[] arrB = new int[]{7,7,7,7,7};

        int startA = 0;
        int startB = 3;
        int length = 2;

        // original implementation
        System.arraycopy(arrA, startA, arrB, startB, length);
        System.out.println(Arrays.toString(arrB));

        // my version
        int[] result = customArrayCopy(arrA, startA, arrB, startB, length);
        System.out.println(Arrays.toString(result));
    }


    public static int[] customArrayCopy(int[] fromArr, int startFrom, int[] toArr, int startTo, int size) {
        for (int i = 0; i < size; i++) {
            toArr[startTo + i] = fromArr[startFrom + i];
        }

        return toArr;
    }
}
