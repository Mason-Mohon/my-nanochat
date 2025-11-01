DEFINITIONS
- Pair: a tuple (int, int)
- Word: an object that stores a list of token IDs (ints) named ids
- MergeJob: a heap item with fields:
    pair: Pair
    count: integer (frequency)
    pos: set<int>  // indices of words where this pair may occur
- Tokenizer:
    merges: map<Pair, int>      // learned pair -> new token id
    pattern: string             // regex text
    compiled_pattern: Regex     // compiled regex

NOTES
- Regex pattern splits text into chunks (GPT-4 style default shown in source).
- Base vocabulary = 256 single-byte tokens (IDs 0..255). Learned merges start at ID 256.
- Training uses Byte Pair Encoding (BPE): iteratively merge most frequent adjacent token pairs.
- Parallel steps can be implemented with threads or a map-reduce style; serial is acceptable but slower.

------------------------------------------------------------
WORD HELPERS
------------------------------------------------------------
Word.new(ids_list):
    w = new Word
    w.ids = ids_list
    return w

Word.pairs():
    // Iterate adjacent pairs from ids
    for i in 0 .. len(ids)-2:
        yield (ids[i], ids[i+1])

Word.merge_pair(pair (a,b), new_id):
    // Merge all NON-OVERLAPPING occurrences of (a,b) -> new_id
    // Return a list of (Pair, delta_count) describing local pair-count changes
    n = len(ids)
    if n < 2: return empty list

    out = empty list<int>
    deltas = empty list<(Pair, int)>
    i = 0
    while i < n:
        if i+1 < n AND ids[i] == a AND ids[i+1] == b:
            left  = last element of out if any, else NONE
            right = ids[i+2] if i+2 < n else NONE

            // old pairs removed and new pairs created around the merge site
            if left != NONE:
                deltas.append(((left, a), -1))
                deltas.append(((left, new_id), +1))
            deltas.append(((a, b), -1))
            if right != NONE:
                deltas.append(((b, right), -1))
                deltas.append(((new_id, right), +1))

            out.push(new_id)  // write merged token
            i = i + 2         // skip a and b
        else:
            out.push(ids[i])
            i = i + 1

    ids = out
    return deltas

------------------------------------------------------------
HEAP ORDERING FOR MERGEJOB
------------------------------------------------------------
Compare(MergeJob x, MergeJob y):
    // Max-heap by count; tie-break by ascending pair for determinism
    if x.count != y.count:
        return x.count > y.count
    else:
        return x.pair < y.pair   // ascending pair

------------------------------------------------------------
COUNT PAIRS (POSSIBLY PARALLEL)
------------------------------------------------------------
count_pairs(words: list<Word>, counts: list<int>):
    // counts[i] is the frequency (weight) of words[i]
    pair_counts = map<Pair, int> (default 0)
    where_to_update = map<Pair, set<int>>  // which words contain the pair

    // Optionally parallel over words
    for each index i, word w in words:
        if len(w.ids) >= 2 AND counts[i] != 0:
            for each (a,b) in w.pairs():
                pair_counts[(a,b)] += counts[i]
                where_to_update[(a,b)].insert(i)

    return (pair_counts, where_to_update)

------------------------------------------------------------
TOKENIZER CORE TRAINING (INCREMENTAL BPE)
------------------------------------------------------------
Tokenizer.train_core_incremental(words, counts, vocab_size):
    assert vocab_size >= 256
    num_merges = vocab_size - 256
    merges.clear()

    // Initial statistics
    (pair_counts, where_to_update) = count_pairs(words, counts)

    // Build a max-heap of MergeJob by (count, then pair)
    heap = empty max-heap<MergeJob>
    for each (pair, pos_set) in where_to_update:
        c = pair_counts.get(pair, 0)
        if c > 0:
            heap.push(MergeJob{ pair=pair, count=c, pos=pos_set })

    merges_done = 0
    while merges_done < num_merges:
        top = heap.pop() if any else BREAK

        // Lazy refresh: if heap count is stale, refresh and reinsert if still > 0
        current = pair_counts.get(top.pair, 0)
        if top.count != current:
            top.count = current
            if top.count > 0: heap.push(top)
            CONTINUE
        if top.count == 0:
            BREAK

        // Assign new token id for this merge
        new_id = 256 + merges_done
        merges[top.pair] = new_id

        // Apply merge to all words where it may occur
        local_pos_updates = map<Pair, set<int>>()
        for each word_idx in top.pos:
            deltas = words[word_idx].merge_pair(top.pair, new_id)

            // Update global pair counts with this word’s weight
            for each (pair, delta) in deltas:
                delta_total = delta * counts[word_idx]
                if delta_total != 0:
                    pair_counts[pair] += delta_total
                    if delta > 0:
                        local_pos_updates[pair].insert(word_idx)

        // Push updated pairs back into heap
        for each (pair, pos_set) in local_pos_updates:
            cnt = pair_counts.get(pair, 0)
            if cnt > 0:
                heap.push(MergeJob{ pair=pair, count=cnt, pos=pos_set })

        merges_done += 1

    // training finished; merges map now defines the learned BPE

------------------------------------------------------------
PUBLIC TOKENIZER METHODS
------------------------------------------------------------
Tokenizer.new():
    t = new Tokenizer
    t.merges = empty map<Pair, int>
    t.pattern = ""          // will be set on train
    t.compiled_pattern = compile_regex("")  // placeholder
    return t

Tokenizer.train_from_iterator(iterator, vocab_size, buffer_size, pattern_opt):
    // pattern_opt is optional; use default GPT-4 pattern if absent
    pattern_str = pattern_opt if provided else DEFAULT_GPT4_PATTERN
    compiled = compile_regex(pattern_str)

    self.pattern = pattern_str
    self.compiled_pattern = compiled

    // Stream ingestion:
    // Read strings in batches of size <= buffer_size,
    // split each string into regex matches, count chunks globally.
    global_counts = map<string, int> (default 0)

    buffer = empty list<string>

    loop:
        exhausted = refill_buffer_from_iterator(iterator, buffer, buffer_size)
        if buffer is empty AND exhausted: BREAK

        // Optionally parallel: for each string, split by regex and count chunks locally, then reduce
        local_counts = map<string, int>()
        for each s in buffer:
            for each match in compiled.find_iter(s):
                piece = match.text
                local_counts[piece] += 1

        // Merge local into global
        for each (k, v) in local_counts:
            global_counts[k] += v

        if exhausted: BREAK

    // Materialize words and counts arrays
    words = []
    counts = []
    for each (chunk_string, c) in global_counts:
        // represent chunk as sequence of raw bytes mapped to IDs 0..255
        byte_ids = [byte_value(u8) as int for each byte in chunk_string]
        words.push( Word.new(byte_ids) )
        counts.push(c)

    // Learn merges
    self.train_core_incremental(words, counts, vocab_size)

Tokenizer.get_pattern():
    return self.pattern

Tokenizer.get_mergeable_ranks():
    // Return (token_bytes, token_id) for all base bytes and learned merges, ordered by token id

    token_bytes = array where token_bytes[i] holds the byte sequence for token id i
    initialize token_bytes[0..255] = [[i]]  // one-byte sequences

    results = []
    for i in 0..255:
        results.push( (token_bytes[i], i) )

    // Sort merges by token id ascending
    sorted_merges = self.merges entries sorted by merged_id asc

    for each (pair=(left,right), merged_id) in sorted_merges:
        merged = token_bytes[left] concatenated with token_bytes[right]
        ensure token_bytes has size > merged_id
        token_bytes[merged_id] = merged
        results.push( (merged, merged_id) )

    return results

Tokenizer.encode(text):
    output_ids = []

    // Split text by compiled pattern into chunks
    for each match in self.compiled_pattern.find_iter(text):
        chunk = match.text

        // Convert chunk to byte-level IDs
        ids = [byte_value(u8) as int for each byte in chunk]

        // Iteratively apply merges greedily by rank (lower token id = higher priority)
        while len(ids) >= 2:
            best = NONE  // (index, pair, new_id) with smallest new_id among available merges
            for i in 0 .. len(ids)-2:
                p = (ids[i], ids[i+1])
                if p in self.merges:
                    new_id = self.merges[p]
                    if best == NONE OR new_id < best.new_id:
                        best = (i, p, new_id)

            if best == NONE:
                BREAK
            (idx, _, new_id) = best
            ids[idx] = new_id
            remove ids[idx+1]

        append ids to output_ids

    return output_ids

------------------------------------------------------------
MODULE EXPORT (if applicable in your host language)
------------------------------------------------------------
export Tokenizer
