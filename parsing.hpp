#include <stdint.h>
#include <array>
#include <string>
#include <sstream>

template<unsigned N>
std::string generic_to_rle(auto&& cellchar, bool flushtrailing = false) {
  std::stringstream result;

  unsigned eol_count = 0;

  for (unsigned j = 0; j < N; j++) {
    char last_val = cellchar(0, j);
    unsigned run_count = 0;

    for (unsigned i = 0; i < N; i++) {
      char val = cellchar(i, j);

      // Flush linefeeds if we find a live cell
      if (val != '.' && val != 'b' && eol_count > 0) {
        if (eol_count > 1)
          result << eol_count;

        result << "$";

        eol_count = 0;
      }

      // Flush current run if val changes
      if (val != last_val) {
        if (run_count > 1)
          result << run_count;
        result << last_val;
        run_count = 0;
      }

      run_count++;
      last_val = val;
    }

    // Flush run of live cells at end of line
    if (last_val != '.' && last_val != 'b') {
      if (run_count > 1)
        result << run_count;

      result << last_val;

      run_count = 0;
    }

    eol_count++;
  }

  // Flush trailing linefeeds
  if (flushtrailing && eol_count > 0) {
    if (eol_count > 1)
      result << eol_count;

    result << "$";
  }

  result << "!";

  return result.str();
}

template<typename T> T generic_parse_rle(const std::string &rle, auto&& interpretcell) {
  std::string noheader;
  std::istringstream iss(rle);

  for (std::string line; std::getline(iss, line); ) {
    if(line[0] != 'x')
      noheader += line;
  }

  T result = {0};

  int cnt = 0;
  int x = 0;
  int y = 0;

  for (char const ch : noheader) {
    if (ch >= '0' && ch <= '9') {
      cnt *= 10;
      cnt += (ch - '0');
    } else if (ch == '$') {
      if (cnt == 0)
        cnt = 1;

      if (cnt == 129)
        // TODO: error
        return result;

      y += cnt;
      x = 0;
      cnt = 0;
    } else if (ch == '!') {
      break;
    } else if (ch == '\r' || ch == '\n' || ch == ' ') {
      continue;
    } else {
      if (cnt == 0)
        cnt = 1;

      for (int j = 0; j < cnt; j++) {
        interpretcell(result, ch, x, y);
        x++;
      }

      cnt = 0;
    }
  }
  return result;
}

template<unsigned W>
std::conditional_t<W == 64, std::array<uint64_t, 64>, std::array<uint32_t, 32>> parse_rle(const std::string &rle) {
  using ArrayType = std::conditional_t<W == 64, std::array<uint64_t, 64>, std::array<uint32_t, 32>>;
  return generic_parse_rle<ArrayType>(rle, [&](ArrayType &result, char ch, int x, int y) -> void {
    if (ch == 'o') {
      if constexpr (W == 64) {
        result[y] |= (1ULL << x);
      } else {
        result[y] |= (1U << x);
      }
    }
  });
}

template<unsigned N, unsigned W>
std::string to_rle(const std::conditional_t<W == 64, std::array<uint64_t, 64>, std::array<uint32_t, 32>> &board) {
  return generic_to_rle<N>([&](int x, int y) -> char {
    if constexpr (W == 64) {
      return (board[y] & (1ULL << x)) ? 'o' : 'b';
    } else {
      return (board[y] & (1U << x)) ? 'o' : 'b';
    }
  });
}


template<unsigned W>
std::pair<std::conditional_t<W == 64, std::array<uint64_t, 64>, std::array<uint32_t, 32>>, 
          std::conditional_t<W == 64, std::array<uint64_t, 64>, std::array<uint32_t, 32>>> 
parse_rle_history(const std::string &rle) {
  using ArrayType = std::conditional_t<W == 64, std::array<uint64_t, 64>, std::array<uint32_t, 32>>;
  ArrayType knownOn = {};
  ArrayType knownOff = {};
  generic_parse_rle<ArrayType>(rle, [&](ArrayType &result, char ch, int x, int y) -> void {
    if (ch == 'A') {
      if constexpr (W == 64) {
        knownOn[y] |= (1ULL << x);
      } else {
        knownOn[y] |= (1U << x);
      }
    }
    if (ch == 'D') {
      if constexpr (W == 64) {
        knownOff[y] |= (1ULL << x);
      } else {
        knownOff[y] |= (1U << x);
      }
    }
  });
  return {knownOn, knownOff};
}

template<unsigned N, unsigned W>
std::string to_rle_history(const std::conditional_t<W == 64, std::array<uint64_t, 64>, std::array<uint32_t, 32>> &knownOn, 
                           const std::conditional_t<W == 64, std::array<uint64_t, 64>, std::array<uint32_t, 32>> &knownOff) {
  return generic_to_rle<N>([&](int x, int y) -> char {
    bool isKnownOn, isKnownOff;
    if constexpr (W == 64) {
      isKnownOn = knownOn[y] & (1ULL << x);
      isKnownOff = knownOff[y] & (1ULL << x);
    } else {
      isKnownOn = knownOn[y] & (1U << x);
      isKnownOff = knownOff[y] & (1U << x);
    }
    if(isKnownOn && isKnownOff) return 'F';
    if(isKnownOn) return 'A';
    if(isKnownOff) return 'D';
    return '.';
  });
}
