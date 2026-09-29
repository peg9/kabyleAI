import com.kabyleai.app.UnigramTokenizer;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Compare UnigramTokenizer au tokenizer de référence.
 *
 * Usage : java TokenizerCheck nllb_config.txt nllb_vocab.tsv reference.tsv
 * reference.tsv : une ligne "texte TAB identifiants séparés par des espaces"
 * (sauts de ligne et tabulations du texte écrits \n et \t), produite par
 * tokens_ref.py.
 */
public class TokenizerCheck {
    public static void main(String[] args) throws Exception {
        int unk = 3;
        Set<Integer> specials = new HashSet<>();
        for (String line : Files.readAllLines(Paths.get(args[0]), StandardCharsets.UTF_8)) {
            if (line.startsWith("unk_id=")) {
                unk = Integer.parseInt(line.substring(7).trim());
            } else if (line.startsWith("specials=")) {
                for (String part : line.substring(9).split(",")) {
                    if (!part.trim().isEmpty()) {
                        specials.add(Integer.parseInt(part.trim()));
                    }
                }
            }
        }
        UnigramTokenizer tokenizer = new UnigramTokenizer(new File(args[1]), specials, unk);

        List<String> lines = Files.readAllLines(Paths.get(args[2]), StandardCharsets.UTF_8);
        int same = 0;
        for (String line : lines) {
            String[] parts = line.split("\t", -1);
            String text = parts[0].replace("\\n", "\n").replace("\\t", "\t");
            int[] want = parts[1].trim().isEmpty() ? new int[0]
                    : Arrays.stream(parts[1].trim().split(" ")).mapToInt(Integer::parseInt).toArray();
            int[] got = tokenizer.encode(text);
            if (Arrays.equals(got, want)) {
                same++;
            } else {
                System.out.println("DIFFERENT : [" + parts[0] + "]");
                System.out.println("  attendu " + Arrays.toString(want));
                System.out.println("  obtenu  " + Arrays.toString(got));
            }
        }
        System.out.println(same + "/" + lines.size() + " phrases tokenisées comme la référence.");
        if (same != lines.size()) {
            System.exit(1);
        }
    }
}
