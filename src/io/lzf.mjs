/**
 * Decompressor for the LZF variant used by PCL's `binary_compressed` PCD data.
 *
 * Port of `lzf_decompress` from liblzf. The stream is a sequence of control
 * bytes: values below 32 introduce a literal run, everything else is a back
 * reference into the output produced so far.
 */
export function lzfDecompress(input, expectedLength) {
  const out = Buffer.alloc(expectedLength);
  let ip = 0;
  let op = 0;

  while (ip < input.length) {
    let ctrl = input[ip++];

    if (ctrl < 32) {
      // Literal run of ctrl + 1 bytes.
      ctrl++;
      if (op + ctrl > out.length) {
        throw new Error("LZF: literal run overflows the output buffer");
      }
      input.copy(out, op, ip, ip + ctrl);
      op += ctrl;
      ip += ctrl;
    } else {
      // Back reference: length lives in the top 3 bits, distance in the rest.
      let len = ctrl >> 5;
      let ref = op - ((ctrl & 0x1f) << 8) - 1;
      if (len === 7) len += input[ip++];
      ref -= input[ip++];

      if (ref < 0) throw new Error("LZF: back reference before start of output");
      len += 2;
      if (op + len > out.length) {
        throw new Error("LZF: back reference overflows the output buffer");
      }
      // Byte-by-byte, because runs are allowed to overlap the write cursor.
      for (let i = 0; i < len; i++) out[op++] = out[ref++];
    }
  }

  if (op !== expectedLength) {
    throw new Error(`LZF: expected ${expectedLength} bytes, produced ${op}`);
  }
  return out;
}
