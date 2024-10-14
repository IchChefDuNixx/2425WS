import java.util.Arrays;

public class Intersection {

    public static void main(String[] args) {

        int[] set1a = new int[]{0,1,2,3,4,5};
        int[] set1b = new int[]{3,4,5,6,7,8};
        int[] set2a = new int[]{0,1,2,3};
        int[] set2b = new int[]{4,5,6};
        int[] set3a = new int[]{0,1,2};
        int[] set3b = new int[]{0,1,2};

        System.out.println(Arrays.toString(intersection(set1a,set1b)));
        System.out.println(Arrays.toString(intersection(set2a,set2b)));
        System.out.println(Arrays.toString(intersection(set3a,set3b)));
    }


    public static int[] intersection(int[] arr1, int[] arr2) {
        // count intersections
        int numIntersections = 0;
        for (int i = 0; i < arr1.length; i++) {
            for (int j = 0; j < arr2.length; j++) {
                if (arr1[i] == arr2[j]) {
                    numIntersections++;
                }
            }
        }

        // compute result
        int[] result = new int[numIntersections];
        int k = 0;
        for (int i = 0; i < arr1.length; i++) {
            for (int j = 0; j < arr2.length; j++) {
                if (arr1[i] == arr2[j]) {
                    result[k] = arr1[i];
                    k++;
                }
            }
        }

        return result;
    }

}
