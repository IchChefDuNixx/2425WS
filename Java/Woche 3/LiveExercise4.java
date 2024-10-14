public class LiveExercise4 {

    public static void main(String[] args) {

        int x = 1;
        int y = 5;
        int z = 3;

        int number = add(x, y, z);

        System.out.println(number);

    }

    public static int add(int a, int b, int c) {
        int sum = a + b + c;
        return sum * sum;
    }


}
