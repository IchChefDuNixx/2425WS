import java.util.Scanner;

public class Maximum2 {
    public static void main(String[] args){

        Scanner scanner = new Scanner(System.in);

        double[] zahlen2 = new double[Integer.parseInt(scanner.nextLine())];

        for(int i = 0; i <= zahlen2.length - 1; i++ ){
            zahlen2[i] = Math.random();
        }
       // max(zahlen2);

        System.out.println("Random Zahl: " + zahlen2);
        System.out.println("Der Max Index ist: " +  max(zahlen2));
    }


    public static double max(double[] zahlen2){
        double maxWert = 0;
        int maxIndex = 0;
        for (int i = 0; i < zahlen2.length; i++) {
            if(maxWert < zahlen2[i]) {
                maxWert = zahlen2[i];
                maxIndex = i;
            }
            System.out.println("Die größte Zahl ist: " + maxWert + " Die nächste Zahl ist: " + zahlen2[i]);
        }

        return maxIndex;
    }
}