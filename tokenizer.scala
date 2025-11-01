import scala.util.matching.Regex

// GPT-4 style pattern for tokenizing text
val GPT_4_PATTERN: Regex = """'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""".r

type Pair = (Int, Int)

// Byte pair encoding (BPE) - iteratively merge most frequent adjacent token pairs
class BytePairEncodingTokenizer(val merges: Map[Pair, Int]) {
    def train(words: List[Word], counts: List[Int]): Unit = {
        // TODO: Implement training
    }
}

class Word(val ids: List[Int]) {
    def merge_pair(pair: Pair, new_id: Int): List[(Pair, Int)] = {
        // Merge all NON-OVERLAPPING occurrences of (a,b) -> new_id
        // Return a list of (Pair, delta_count) describing local pair-count changes
        val n = ids.length
        if (n < 2) return List()
        val out = List[Int]()
        val deltas = List[(Pair, Int)]()
        var i = 0
        while (i < n) {
            if (i+1 < n && ids[i] == pair._1 && ids[i+1] == pair._2) {
                val left = if (out.nonEmpty) out.last else None
                val right = if (i+2 < n) ids[i+2] else None
                if (left != None) {
                    deltas.append(((left, pair._1), -1))
                    deltas.append(((left, new_id), 1))
                }
                deltas.append(((pair._1, pair._2), -1))
                if (right != None) {
                    deltas.append(((pair._2, right), -1))
                    deltas.append(((new_id, right), 1))
                }
                out.append(new_id)
                i = i + 2
            } else {
                out.append(ids[i])
                i = i + 1
            }
        }
        ids = out
        deltas
    }
}