import chess

ranks = [8, 7, 6, 5, 4, 3, 2, 1]
verticals = ["a", "b", "c", "d", "e", "f", "g", "h"]


def non_trivial_attack(attack_fields, row, column, chess_board, row_change, column_change):
    new_row = row_change + row
    new_column = column + column_change
    while 0 <= new_row <= 7 and 0 <= new_column <= 7:
        attack_fields[new_row][new_column] = 1
        if chess_board[new_row][new_column] != 0:
            break
        new_row += row_change
        new_column += column_change
    return attack_fields


def squares_of_attack(chess_board, chess_peace, row, column):
    attack_fields = [[0]*8, [0]*8, [0]*8, [0]*8, [0]*8, [0]*8, [0]*8, [0]*8]
    print(chess_peace, verticals[column], ranks[row], ": ", sep="", end="")
    peace_name = chess_peace[0]


    if peace_name == "P": # pawn
        if chess_peace[-1] == "*":
            new_row = row + 1 # black pawn attacks downwards
        else:
            new_row = row - 1 # white pawn attacks forward

        if column != 0 and column != 7:
            attack_fields[new_row][column + 1] = 1
            attack_fields[new_row][column - 1] = 1
        elif column == 0:
            attack_fields[new_row][column + 1] = 1
        elif column == 7:
            attack_fields[new_row][column - 1] = 1


    elif peace_name == "N": # knight
        rows_change = [1, -1, 2, -2]
        column_change = [1, -1, 2, -2]
        for i in rows_change:
            for j in column_change:
                new_row = row + i
                new_column = column + j
                if 0 <= new_column <= 7 and 0 <= new_row <= 7 and \
                        not (abs(i) == abs(j)):
                    attack_fields[new_row][new_column] = 1


    elif peace_name == "K": #king
        rows_change = [1, 0, -1]
        column_change = [1, 0, -1]
        for i in rows_change:
            for j in column_change:
                new_row = row + i
                new_column = column + j
                if 0 <= new_column <= 7 and 0 <= new_row <= 7 and \
                        not (i == 0 and j == 0):
                    attack_fields[new_row][new_column] = 1


    elif peace_name == "R": #rook
        rows_change = [1, -1]
        column_change = [1, -1]
        for i in rows_change:
            attack_fields = non_trivial_attack(attack_fields, row,
                                               column, chess_board,
                                               i, 0)
        for j in column_change:
            attack_fields = non_trivial_attack(attack_fields,
                                               row, column, chess_board,
                                               0, j)


    elif peace_name == "B": #bishop
        rows_change = [1, -1]
        column_change = [-1, 1]
        for i in rows_change:
            for j in column_change:
                attack_fields = non_trivial_attack(attack_fields, row,
                                                   column, chess_board, i, j)


    elif peace_name== "Q": #queen
        rows_change = [-1, 0, 1]
        column_change = [-1, 0, 1]
        for i in rows_change:
            for j in column_change:
                if not (i == 0 and j == 0):
                    attack_fields = non_trivial_attack(attack_fields, row,
                                                   column, chess_board, i, j)

    else:
        return 0

    for i in range(8):
        for j in range(8):
            if attack_fields[i][j] == 1:
                print(verticals[j], ranks[i], sep='', end=', ')
    print()
    return attack_fields



chess1 = [
    ["R*", "N*", "B*", "Q*", "K*", "B*", "N*", "R*"], # 8
    ["P*", "P*", "P*", "P*", "P*", "P*", "P*", "P*"], # 7
    [ 0,    0,    0,    0,    0,    0,    0,    0  ], # 6
    [ 0,    0,    0,    0,    0,    0,    0,    0  ], # 5
    [ 0,    0,    0,    0,    0,    0,    0,    0  ], # 4
    [ 0,    0,    0,    0,    0,    0,    0,    0  ], # 3
    ["P",  "P",  "P",  "P",  "P",  "P",  "P",  "P" ], # 2
    ["R",  "N",  "B",  "Q",  "K",  "B",  "N",  "R" ]  # 1
]#    a     b     c     d     e     f     g     h

graph_of_attacks = []
print("chess1")
for i in range(8):
    for j in range(8):
        if chess1[i][j] != 0:
            graph_of_attacks.append(squares_of_attack(chess1, chess1[i][j], i, j))

chess2 = [
    [ 0,    0,    0,    0,    0,    0,   "K*",  0  ], # 8
    [ 0,    0,    0,    0,    0,   "Q*",  0,    0  ], # 7
    [ 0,    0,    0,    0,    0,    0,   "Q",  "P" ], # 6
    [ 0,    0,    0,    0,    0,    0,   "P",  "K" ], # 5
    [ 0,    0,    0,    0,    0,    0,    0,    0  ], # 4
    [ 0,    0,    0,    0,    0,    0,    0,    0  ], # 3
    [ 0,    0,    0,    0,    0,    0,    0,    0  ], # 2
    [ 0,    0,    0,    0,    0,    0,    0,    0  ], # 1
]#    a     b     c     d     e     f     g     h

print("\nchess2")
for i in range(8):
    for j in range(8):
        if chess2[i][j] != 0:
            graph_of_attacks.append(squares_of_attack(chess2, chess2[i][j], i, j))


print(chess.Board())