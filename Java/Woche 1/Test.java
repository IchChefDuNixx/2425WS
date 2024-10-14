public class Test {

    public static void main(String[] args) {

        System.out.println("Hello World");

        int[] y = new int[5];
        y[0] = 12;  //Array mit Werten füllen
        y[1] = 7;
        y[2] = 17;
        y[3] = -2;
        y[4] = 0;


        for(int i=4; i>=0; i--) {
            System.out.println(y[i]);
        }

        System.out.println("---");

        for(int i=4; i>-1; i--) {
            System.out.println(y[i]);
        }

    }

}
