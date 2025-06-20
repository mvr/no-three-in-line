#include <stdint.h>
#include <array>
#include <string>
#include <sstream>

std::string generic_to_rle(auto&& cellchar, bool flushtrailing = false) {
  std::stringstream result;

  unsigned eol_count = 0;

  for (unsigned j = 0; j < 64; j++) {
    char last_val = cellchar(0, j);
    unsigned run_count = 0;

    for (unsigned i = 0; i < 64; i++) {
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

std::array<uint64_t, 64> parse_rle(const std::string &rle) {
  return generic_parse_rle<std::array<uint64_t, 64>>(rle, [&](std::array<uint64_t, 64> &result, char ch, int x, int y) -> void {
    if (ch == 'o') {
      result[y] |= (1ULL << x);
    }
  });
}

std::string to_rle(const std::array<uint64_t, 64> &board) {
  return generic_to_rle([&](int x, int y) -> char {
    return (board[y] & (1ULL << x)) ? 'o' : 'b';
  });
}


std::pair<std::array<uint64_t, 64>, std::array<uint64_t, 64>> parse_rle_history(const std::string &rle) {
  std::array<uint64_t, 64> knownOn = {0};
  std::array<uint64_t, 64> knownOff = {0};
  generic_parse_rle<std::array<uint64_t, 64>>(rle, [&](std::array<uint64_t, 64> &result, char ch, int x, int y) -> void {
    if (ch == 'A')
      knownOn[y] |= (1ULL << x);
    if (ch == 'D')
      knownOff[y] |= (1ULL << x);
  });
  return {knownOn, knownOff};
}

std::string to_rle_history(const std::array<uint64_t, 64> &knownOn, const std::array<uint64_t, 64> &knownOff) {
  return generic_to_rle([&](int x, int y) -> char {
    bool isKnownOn = knownOn[y] & (1ULL << x);
    bool isKnownOff = knownOff[y] & (1ULL << x);
    if(isKnownOn && isKnownOff) return 'F';
    if(isKnownOn) return 'A';
    if(isKnownOff) return 'D';
    return '.';
  });
}
