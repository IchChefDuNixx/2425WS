public class Coffeeshop {
    public static void main(String[] args) {
        Cappuccino order1 = new Cappuccino("strong", null, true);
        LatteMacchiato order2 = new LatteMacchiato("medium", "vanilla syrup", false);
        Espresso order3 = new Espresso("weak", "sugar", false);
        Cappuccino order4 = new Cappuccino("medium", "sweetener", true);

        double revenue = order1.price + order2.price + order3.price + order4.price;
        System.out.println(revenue);

        int numMilk = order1.foamedMilk + order2.foamedMilk + order3.foamedMilk + order4.foamedMilk;
        System.out.println(numMilk);

        int numPaperCups = 0;
        if (order1.container == "paper cup") { numPaperCups++; }
        if (order2.container == "paper cup") { numPaperCups++; }
        if (order3.container == "paper cup") { numPaperCups++; }
        if (order4.container == "paper cup") { numPaperCups++; }

        System.out.println(numPaperCups);
    }
}

class Espresso {
    final int foamedMilk = 0;
    double price = 1.50;
    String strength;
    boolean sweetened;
    String sweetener;
    boolean toGo;
    String container; // could be extra classes

    public Espresso(String strength, String sweetener, boolean toGo) {
        this.strength = strength;
        if (sweetener != null) {
            this.sweetener = sweetener;
            this.sweetened = true;
            this.price += 0.20;
        } else {
            this.sweetener = "";
            this.sweetened = false;
        }
        if (toGo) {
            this.container = "paper cup";
            this.price += 0.10;
        } else {
            this.container = "porcelain cup";
        }
    }
}

class Cappuccino {
    final int foamedMilk = 1;
    double price = 2.00;
    String strength;
    boolean sweetened;
    String sweetener;
    boolean toGo;
    String container;

    public Cappuccino(String strength, String sweetener, boolean toGo) {
        this.strength = strength;
        if (sweetener != null) {
            this.sweetener = sweetener;
            this.sweetened = true;
            this.price += 0.20;
        } else {
            this.sweetener = "";
            this.sweetened = false;
        }
        if (toGo) {
            this.container = "paper cup";
            this.price += 0.10;
        } else {
            this.container = "porcelain cup";
        }
    }
}

class LatteMacchiato {
    final int foamedMilk = 2;
    double price = 2.50;
    String strength;
    boolean sweetened;
    String sweetener;
    boolean toGo;
    String container;

    public LatteMacchiato(String strength, String sweetener, boolean toGo) {
        this.strength = strength;
        if (sweetener != null) {
            this.sweetener = sweetener;
            this.sweetened = true;
            this.price += 0.20;
        } else {
            this.sweetener = "";
            this.sweetened = false;
        }
        if (toGo) {
            this.container = "paper cup";
            this.price += 0.10;
        } else {
            this.container = "glass";
        }
    }
}

class AnisFenchelKümmelTea {
    final int foamedMilk = 0;
    double price = 1.99;
    String strength;
    boolean sweetened;
    String sweetener;
    boolean toGo;
    String container;

    public AnisFenchelKümmelTea(String strength, String sweetener, boolean toGo) {
        this.strength = strength;
        if (sweetener != null) {
            this.sweetener = sweetener;
            this.sweetened = true;
            this.price += 0.20;
        } else {
            this.sweetener = "";
            this.sweetened = false;
        }
        if (toGo) {
            this.container = "paper cup";
            this.price += 0.10;
        } else {
            this.container = "porcelain cup";
        }
    }
}

