public class PasswordChecker {
    public static boolean isGoodPassword(String pwd) {
        if (pwd == null || pwd.length() < 8) {
            return false;
        }
        int numbers = 0, upper = 0, lower = 0, special = 0;
        for (int i = 0; i < pwd.length(); i++) {
            if (Character.isDigit(pwd.charAt(i))) {
                numbers++;
                continue;
            }
            if (Character.isLowerCase(pwd.charAt(i))) {
                lower++;
                continue;
            }
            if (Character.isUpperCase(pwd.charAt(i))) {
                upper++;
                continue;
            }
            special++;
        }

        return (numbers >= 2) && (upper >= 1) && (lower >= 1) && (special >= 1);
    }

    public static void main(String[] args) {
        System.out.print(isGoodPassword(null));

    }
}

