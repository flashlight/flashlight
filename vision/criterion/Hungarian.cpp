#include <limits>
#include <assert.h>

void findUncoveredZero(float* costs, int* colCover, int* rowCover, int nrows, int ncols, 
    int* row, int* col) {
  bool done = false;
  *row = -1;
  *col = -1;
  for(int c = 0; c < ncols && !done; c++) {
    for(int r = 0; r < nrows && !done; r++) {
      const float cost = costs[c * nrows + r];
      if (cost == 0 && colCover[c] == 0 && rowCover[r] == 0) {
        *row = r;
        *col = c;
        done = true;
      }
    }
  }
}

bool isStarInRow(int* marks, int row, int nrows, int ncols) {
  for(int c = 0; c < ncols; c++) {
    if(marks[c * nrows + row] == 1) {
      return true;
    };
  }
  return false;
}

int findStarInRow(int* marks, int row, int nrows, int ncols) {
  for(int c = 0; c < ncols; c++) {
    if(marks[c * nrows + row] == 1) {
      return c;
    };
  }
  return -1;
}

// M x N matrix M = nrows, N = ncols
void step_one(float* costs, const int nrows, const int ncols) {
  for (int r = 0; r < nrows; r++) {
    float min_val = std::numeric_limits<float>::max();
    for (int c = 0; c < ncols; c++) {
      float val = costs[c * nrows + r];
      if (val < min_val) {
        min_val = val;
      }
    }
    for (int c = 0; c < ncols; c++) {
      costs[c * nrows + r] -= min_val;
    }
  }
}

// Iterate through rows, and mark 0s if they have not already been covered by a previous marking
void step_two(float* costs, int* marks, int* colCover, int* rowCover, const int nrows, const int ncols) {
  for(int r = 0; r < nrows; r++) {
    for(int c = 0; c < ncols; c++) {
      float cost = costs[c * nrows + r];
      if (cost == 0.0 && rowCover[r] == 0 && colCover[c] == 0) {
        marks[c * nrows + r] = 1;
        rowCover[r] = 1;
        colCover[c] = 1;
      }
    }
  }
  for(int r = 0; r < nrows; r++) {
    rowCover[r] = 0;
  }
  for(int c = 0; c < ncols; c++) {
    colCover[c] = 0;
  }
}

int step_three(int* marks, int* colCover, int* rowCover, int nrows, int ncols) {
  for(int r = 0; r < nrows; r++) {
    for(int c = 0; c < ncols; c++) {
      const int mark = marks[c * nrows + r];
      if (mark == 1) {
        colCover[c] = 1;
      }
    }
  }
  int coveredCols = 0;
  for(int c = 0; c < ncols; c++) {
    coveredCols += colCover[c];
  }
  if(coveredCols == ncols || coveredCols >= nrows) {
    return 7;
  } else {
    return 4;
  }
}

// Find a noncovered zero and "prime it". If there are no uncovered zeros in the row containing
// this zero, go to 5. Otherwise, cover this row, and uncovered column containing starred zero.
// Continue until there are no more uncovered zeros left. Then go to 6. 
int step_four(float* costs, int* marks, int* colCover, int* rowCover, int nrows, int ncols, int* firstPathRow,
    int* firstPathCol) {
  bool done = false;
  while(!done) {
    int row, col;
    findUncoveredZero(costs, colCover, rowCover, nrows, ncols, &row, &col);
    if (row < 0 && col < 0) {
      return 6;
    }
    else {
      // "Prime it"
      marks[col * nrows + row] = 2;
      if(isStarInRow(marks, row, nrows, ncols)) {
        int col = findStarInRow(marks, row, nrows, ncols);
        rowCover[row] = 1;
        colCover[col] = 0;
      } else {
        *firstPathRow = row;
        *firstPathCol = col;
        done = true;
        return 5;
      }
    }
  }
}

int findStarInCol(int* masks, int col, int nrows, int ncols) {
  for(int r = 0; r < nrows; r++) {
    if(masks[col * nrows + r] == 1) {
      return r;
    }
  }
  return -1;
}

int findPrimeInRow(int* masks, int row, int nrows, int ncols) {
  for(int c = 0; c < ncols; c++) {
    if(masks[c * nrows + row] == 2) {
      return c;
    }
  }
  return -1;

}

void augmentPaths(int* paths, int pathCount, int* marks, int nrows, int ncols) {
  for(int p = 0; p < pathCount; p++) {
    int row = paths[p * 2];
    int col = paths[p * 2 + 1];
    if(marks[col * nrows + row] == 1) {
      marks[col * nrows + row] = 0;
    } else {
      marks[col * nrows + row] = 1;
    }
  }
}

void clearCover(int* cover, int n) {
  for(int i = 0; i < n; i++) {
    cover[i] = 0;
  }
}

void erasePrimes(int* marks, int nrows, int ncols) {
  for(int c = 0; c < ncols; c++) {
    for(int r = 0; r < nrows; r++) {
      if(marks[c * nrows + r] == 2) {
        marks[c * nrows + r] = 0;
      }
    }
  }
}

int step_five(float* costs, int* marks, int* colCover, int* rowCover, int* path, int firstPathRow, int firstPathCol, int nrows, int ncols) {
  int r = -1;
  int c = -1;
  int pathCount = 1;
  path[(pathCount - 1) * 2] = firstPathRow;
  path[(pathCount - 1) * 2 + 1] = firstPathCol;
  bool done = false;
  while(!done) {
    r = findStarInCol(marks, path[(pathCount - 1) * 2 + 1], nrows, ncols);
    if(r > -1) {
      pathCount += 1;
      path[(pathCount - 1) * 2] = r;
      path[(pathCount - 1) * 2 + 1] = path[(pathCount - 2) * 2 + 1];
    } else {
      done = true;
    }
    if(!done) {
      c = findPrimeInRow(marks, path[(pathCount - 1) * 2], nrows, ncols);
      pathCount += 1;
      path[(pathCount - 1) * 2] = path[(pathCount - 2) * 2];
      path[(pathCount - 1) * 2 + 1] = c;
    }
  }
  augmentPaths(path, pathCount, marks, nrows, ncols);
  clearCover(colCover, ncols);
  clearCover(rowCover, nrows);
  erasePrimes(marks, nrows, ncols);
  return 3;
}

float findSmallestNotCovered(float* costs, int* colCover, int* rowCover, int nrows, int ncols) {
  float minValue = std::numeric_limits<float>::max();
  for(int c = 0; c < ncols; c++) {
    for(int r = 0; r < nrows; r++) {
      if (colCover[c] == 0 && rowCover[r] == 0) {
        const float cost = costs[c * nrows + r];
        if(cost < minValue) {
          minValue = cost;
        }
      }
    }
  }
  return minValue;
}

int step_six(float* costs, int* marks, int* colCover, int* rowCover, int nrows, int ncols) {
  float minVal = findSmallestNotCovered(costs, colCover, rowCover, nrows, ncols);
  for(int c = 0; c < ncols; c++) {
    for(int r = 0; r < nrows; r++) {
      if(rowCover[r] == 1) {
        costs[c * nrows + r] += minVal;
      }
      if (colCover[c] == 0) {
        costs[c * nrows + r] -= minVal;
      }
    }
  }
  return 4;
}


