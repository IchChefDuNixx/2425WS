public class Library {
    public static void main(String[] args) {
        Book book1 = new Book("Wind and Truth", "Brando Sando", 2024, Genre.Fantasy);
        book1.lendBook();
    }
}

class Book implements BookInterface {
    String title;
    String author;
    int year;
    Genre genre;
    boolean available = true;

    public Book() {
        this("Unknown", "Unknown", 0, Genre.Unknown);
    }

    public Book(String title, String author, int year, Genre genre) {
        this.title = title;
        this.author = author;
        this.year = year;
        this.genre = genre;
    }

    public void lendBook() {
        this.available = false;
    }

    public String getAuthor() {
        return this.author;
    }

    public void setAuthor(String author) {
        this.author = author;
    }

    public Genre getGenre() {
        return this.genre;
    }

    public void setGenre(Genre genre) {
        this.genre = genre;
    }
}

enum Genre {
    Fantasy,
    SciFi,
    Contemporary,
    Romance,
    Thriller,
    Historical,
    Nonfiction,
    Unknown
}