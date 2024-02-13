import chess.pgn


def make_move_and_write_it(move, corresponding_moves, square_figure):
    # фичи
    end_square = chess.square_name(move.to_square)
    start_square = chess.square_name(move.from_square)

    # взятие фигуры
    if end_square in square_figure:
        figure_to_drop = square_figure[end_square]
        corresponding_moves[figure_to_drop].append(0)

    # взятие на проходе
    if chess.square_file(move.from_square)!= \
            chess.square_file(move.to_square) and \
            (square_figure[start_square][0]).lower()=='p' and \
            end_square not in square_figure:
        square_to_drop = (chess.FILE_NAMES[
            chess.square_file(move.from_square)]) + \
                         str(chess.square_rank(move.from_square) + 1)

        figure_to_drop = square_figure[square_to_drop]
        corresponding_moves[figure_to_drop].append(0)
        del square_figure[square_to_drop]

    # рокировкa (для записи движения ладьи)
    if chess.square_distance(move.to_square, move.from_square)==2 \
            and (square_figure[start_square][0]).lower()=='k':

        # длинная
        if chess.square_file(move.to_square)==2:
            square_figure['d' + str(chess.square_rank(move.from_square) + 1)] = \
                square_figure[
                    'a' + str(chess.square_rank(move.from_square) + 1)]

            corresponding_moves[square_figure
            ['a' + str(chess.square_rank(move.from_square) + 1)]] \
                .append(
                'd' + str(chess.square_rank(move.from_square) + 1) + "(O-O-O)")

            del square_figure[
                'a' + str(chess.square_rank(move.from_square) + 1)]

        # короткая
        else:
            square_figure['f' + str(chess.square_rank(move.from_square) + 1)] = \
                square_figure[
                    'h' + str(chess.square_rank(move.from_square) + 1)]

            corresponding_moves[square_figure[
                'f' + str(chess.square_rank(move.from_square) + 1)]] \
                .append(
                'h' + str(chess.square_rank(move.from_square) + 1) + "(O-O)")

            del square_figure[
                'h' + str(chess.square_rank(move.from_square) + 1)]

    # перезапись в словаре клетки/фигуры
    square_figure[end_square] = square_figure[start_square]

    # запись в какую фигуру превратилось (модифицирует end_square)
    if (square_figure[start_square][0]).lower()=='p' and \
            (chess.square_rank(chess.parse_square(start_square))==0 or
             chess.square_rank(move.from_square)==7):
        end_square += move.uci()[-1]

    # добавляем ход в список ходов фигуры
    corresponding_moves[square_figure[start_square]].append(end_square)
    # удаляем предыдущий ход фигуры
    del square_figure[start_square]


chess_file = open('lichess_study_--_--1-0_by_DaniilTonkikh2009_2021.08.29.pgn')

game = chess.pgn.read_game(chess_file)
print(game.headers['Event'])
print(chess_file.tell())

board = game.board()
squares = chess.SQUARES

corresponding_moves = {}  # ходы каждой фигуры за все время
square_figure = {}  # клетка и какая фигура на ней стоит

for square in squares:
    if board.piece_at(square):
        string = str(board.piece_at(square)) + chess.square_name(square)
        square_figure[chess.square_name(square)] = string
        corresponding_moves[string] = []
        corresponding_moves[string].append(chess.square_name(square))

for move in game.mainline_moves():
    print(board, "\n")
    print(square_figure)
    print(move.uci())

    figure_attack = {}
    for square in squares:
        attack = board.attacks(square)
        if attack:
            figure_attack[chess.square_name(square)] = list(
                board.attacks(square))

    make_move_and_write_it(move, corresponding_moves, square_figure)

    board.push(move)

print(corresponding_moves)
