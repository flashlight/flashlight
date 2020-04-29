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
    int* firstPathRow) {
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


