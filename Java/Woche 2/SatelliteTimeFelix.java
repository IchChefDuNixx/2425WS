import java.util.Scanner;

public class SatelliteTimeFelix {
    public static void main(String[] args) {

        Scanner scanner = new Scanner(System.in);
        long Seconds = Long.parseLong(scanner.nextLine());

        // more compact than before
        long days = Seconds / (24 * 60 * 60);
        long hours = (Seconds % (24 * 60 * 60)) / (60 * 60);
        long minutes = (Seconds % (60 * 60)) / 60;
        long seconds = Seconds % 60;

        // a worse alternative to above
        // long days = Seconds / (24 * 60 * 60);
        // long hours = (Seconds - days * (24 * 60 * 60)) / (60 * 60);
        // long minutes = (Seconds - days * (24 * 60 * 60) - hours * (60 * 60)) / 60;
        // long seconds = Seconds - days * (24 * 60 * 60) - hours * (60 * 60) - minutes * 60;

        System.out.println(Seconds + " seconds equals to " + days + " days, " + hours + " hours, " + minutes + " minutes, " + seconds + " seconds.");
    }
}