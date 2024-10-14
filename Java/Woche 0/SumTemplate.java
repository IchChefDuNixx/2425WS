public class SumTemplate {
    public static void main(String[] args) {
        /// Integer which is counted up to
        int boundary;
        // Sum of the previously added digits
        int sum;
        // Integer which comes next
        int counter;
        // Define the starting value of the variables
        boundary = 4;
        sum = -10;
        counter = 1;

        while (counter <= boundary) {
            sum = sum + counter;
            counter = counter + 1;
        }
        
        // Output the sum, together with an explanatory text
        System.out.print("The sum of the digits from 1 to ");
        System.out.print(boundary);
        System.out.print(" is ");
        System.out.println(sum);
        System.out.println(counter);



        // loop 1
        boundary = 4;
        sum = 1;
        counter = 2;

        // loop 2
        boundary = 4;
        sum = 3;
        counter = 3;

        // loop 3
        boundary = 4;
        sum = 6;
        counter = 4;

        // loop 4
        boundary = 4;
        sum = 10;
        counter = 5;




        
        // Strings sind (vereinfacht gesagt) char Arrays
        String a = "aaa";
        char[] b = new char[3];
        char[] c = {'c', 'c', 'c'};





        // Integer Division löscht Rest
        int x = 17;
        int y = 4;

        System.out.println(x/y);
        System.out.println(".");
        System.out.println(x%y/4.0);





        int[] zahlen = new int[10];

        for (int i = 0; i <= 9; i++) {
            zahlen[i] = i;
        }



        // int i = 0;

        // while (i <= 9) {
        //     zahlen[i] = 5;
        // }

/*      // loop 0
        i = 0;
        zahlen = [null, null, null, null, null];

        // loop 1
        i = 1;
        zahlen = [5, null, null, null, null];

        // loop 2
        i = 2;
        zahlen = [5, 5, null, null, null] */

    }
}