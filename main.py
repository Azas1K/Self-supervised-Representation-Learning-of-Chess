import chess.pgn
import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt
import seaborn as sns

import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
from collections import Counter

# plt.style.use('tableau-colorblind10')


def fool_make_move_and_write_it(move, corresponding_moves,
                                square_figure, count_pieces):
    # фичи
    end_square = chess.square_name(move.to_square)
    start_square = chess.square_name(move.from_square)

    # взятие фигуры
    if end_square in square_figure:
        figure_to_drop = square_figure[end_square]
        if figure_to_drop[-1].isalpha():
            count_pieces[figure_to_drop[-1]] -= 1
        else:
            count_pieces[figure_to_drop[0]] -= 1
        corresponding_moves[figure_to_drop].append(0)

    # взятие на проходе
    if chess.square_file(move.from_square)!= \
            chess.square_file(move.to_square) and \
            (square_figure[start_square][0]).lower()=='p' and \
            (len(square_figure[start_square])) == 3 and \
            end_square not in square_figure:
        if (square_figure[start_square][0])=='p':
            square_to_drop = (chess.FILE_NAMES[
                chess.square_file(move.to_square)]) + \
                             str(chess.square_rank(move.to_square) + 2)
        else:
            square_to_drop = (chess.FILE_NAMES[
                chess.square_file(move.to_square)]) + \
                             str(chess.square_rank(move.to_square))

        figure_to_drop = square_figure[square_to_drop]
        count_pieces[figure_to_drop[0]] -= 1
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
                'd' + str(chess.square_rank(move.from_square) + 1)
                # + "(O-O-O)"
            )

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
                'h' + str(chess.square_rank(move.from_square) + 1)
                # + "(O-O)"
            )

            del square_figure[
                'h' + str(chess.square_rank(move.from_square) + 1)]

    # перезапись в словаре клетки/фигуры
    square_figure[end_square] = square_figure[start_square]

    # запись в какую фигуру превратилось (модифицирует end_square)
    if move.uci()[-1].isalpha():

        count_pieces[square_figure[start_square][0]] -= 1
        if square_figure[start_square][0].isupper():
            count_pieces[move.uci()[-1].upper()] += 1
            square_figure[end_square] += move.uci()[-1].upper()
        else:
            count_pieces[move.uci()[-1]] += 1
            square_figure[end_square] += move.uci()[-1]

        corresponding_moves[
            square_figure[end_square]] = corresponding_moves.pop(
            square_figure[start_square])
        square_figure[start_square] = square_figure[end_square]

        end_square += (move.uci()[-1].upper()
                       if square_figure[start_square][0].isupper()
                       else move.uci()[-1])


    # добавляем ход в список ходов фигуры
    corresponding_moves[square_figure[start_square]].append(end_square)
    # удаляем предыдущий ход фигуры
    del square_figure[start_square]


def fool_write_tracking_netework(move, tracking_network, square_figure):
    # фичи
    end_square = chess.square_name(move.to_square)
    start_square = chess.square_name(move.from_square)
    changed_squares = []

    if (square_figure[start_square][0].lower()) == 'n' or \
            square_figure[start_square][-1].lower()=='n':
        tracking_network[move.from_square][move.to_square] = 1
        return

    if chess.square_file(move.to_square)>=chess.square_file(move.from_square):
        changed_files=chess.FILE_NAMES[chess.square_file(move.from_square):
                      chess.square_file(move.to_square)+1]
    else:
        changed_files = chess.FILE_NAMES[chess.square_file(move.to_square):
                                         chess.square_file(move.from_square)+1][::-1]

    if chess.square_rank(move.to_square) >= chess.square_rank(move.from_square):
        changed_ranks = chess.RANK_NAMES[chess.square_rank(move.from_square):
                      chess.square_rank(move.to_square)+1]
    else:
        changed_ranks = chess.RANK_NAMES[chess.square_rank(move.to_square):
                                         chess.square_rank(move.from_square)+1][::-1]

    if len(changed_ranks) == 1:
        for index in range(0, len(changed_files)):
            changed_squares.append(chess.parse_square(changed_files[index]+changed_ranks[0]))
    elif len(changed_files) == 1:
        for index in range(0, len(changed_ranks)):
            changed_squares.append(chess.parse_square(changed_files[0]+changed_ranks[index]))
    else:
        for index in range(0, len(changed_ranks)):
            changed_squares.append(
                chess.parse_square(changed_files[index] + changed_ranks[index]))

    # normal_names = []
    # for k in range(len(changed_squares)):
    #     normal_names.append(chess.square_name(changed_squares[k]))
    # print(normal_names)

    for index in range(1, len(changed_squares)):
        tracking_network[changed_squares[index-1]][changed_squares[index]] = 1


##
# chess_file = open('test.pgn')
#
# game = chess.pgn.read_game(chess_file)
# # print(game.headers['Event'])
# # print(game.end().ply()/2)
#
# board = game.board()
# squares = chess.SQUARES
#
#
# corresponding_moves = {}  # ходы каждой фигуры за все время
# square_figure = {}  # клетка и какая фигура на ней стоит
#
# for square in squares:
#     if board.piece_at(square):
#         string = str(board.piece_at(square)) + chess.square_name(square)
#         square_figure[chess.square_name(square)] = string
#         corresponding_moves[string] = []
#         corresponding_moves[string].append(chess.square_name(square))
#
# game_attack_matrix=[]
# game_singular_values=[]
# game_ranks=[]
#
# for move in game.mainline_moves():
#     # print(board, "\n")
#     # print(square_figure)
#     # print(move.uci())
#     move_attack_matrix = []
#     figure_attack = {}
#
#     for i, figure in enumerate(corresponding_moves.keys()):
#         move_attack_matrix.append([0]*64)
#         square = corresponding_moves[figure][-1]
#         if square != 0:
#             attacked_squares = board.attacks(chess.parse_square(square))
#             for j in attacked_squares:
#                 move_attack_matrix[i][j] = 1

    # for square in squares:
    #     attack = board.attacks(square)
    #     if attack:
    #         figure_attack[chess.square_name(square)] = list(
    #             board.attacks(square))
    # print(figure_attack)

    # #
    # make_move_and_write_it(move, corresponding_moves, square_figure)
    # game_attack_matrix.append(move_attack_matrix)
    # move_singular_values = list(np.linalg.svd(move_attack_matrix)[1])
    # game_singular_values.append(move_singular_values)
    # game_ranks.append(np.linalg.matrix_rank(move_attack_matrix))
    # board.push(move)

# print(game_attack_matrix)
# print(game_singular_values)
# print(game_ranks)
# print(corresponding_moves)

# #
# df = pd.DataFrame(data={'Game_attack_matrix': [game_attack_matrix],
#                         'Game_singular_values': [game_singular_values],
#                         'Game_ranks': [game_ranks]})

# print(df)
# print(np.size(df.iloc[0]['Game_singular_values'][:], 0), np.size(df.iloc[0]['Game_singular_values'], 1))
# print([row[0] for row in df.iloc[0]['Game_singular_values']])
# matrix = df.iloc[0]['Game_singular_values']
# print(matrix)

# for i in range(10):
#     plt.subplot(5, 2, i+1)
#     sns.lineplot(
#         x=range(np.size(df.iloc[0]['Game_singular_values'], 0)),
#         y=[row[i] for row in df.iloc[0]['Game_singular_values']],
#         label=f'{i+1} синг. число')
# plt.xlabel("Полуходы")
# plt.ylabel("Значения синг. числа")
# plt.title('Значения первых 10 сингулярных чисел')
# plt.tight_layout()
# plt.show()

# for i in range(10):
#     sns.lineplot(
#         x=range(np.size(df.iloc[0]['Game_singular_values'], 0)),
#         y=[row[i] for row in df.iloc[0]['Game_singular_values']],
#         label=f'{i+1} синг. число')
# plt.xlabel("Полуходы")
# plt.title('Значения первых 10 сингулярных чисел')
# plt.ylabel("Значения синг. числа")
# plt.show()

# plt.figure(figsize=(20, 10))
# plt.plot(df.iloc[0]['Game_ranks'], color='blue', marker='o', linestyle='dashed',
#      linewidth=1, markersize=5)
# plt.title('Ранг матрицы атаки')
# plt.xlabel("Полуходы")
# plt.ylabel("Ранг")
# plt.show()


squares = chess.SQUARES
files = [f for f in os.listdir('database/')]

games_results = []
games_attack_matrix=[]
games_positional_network = []
games_tracking_network = []
games_features_positional = []
games_features_traking = []
games_shanon = []
count=0

for file in files:
    with open('database/' + file, encoding='unicode_escape') as pgn:
        game = chess.pgn.read_game(pgn)

        while game:

            result = game.headers['Result']
            board = game.board()

            if (game.end().board().is_checkmate()):

                # print(game.headers)
                count_pieces = {'K': 1, 'k': 1, 'Q': 1, 'q': 1,
                                'R': 2, 'r': 2, 'N': 2, 'n': 2,
                                'B': 2, 'b': 2, 'P': 8, 'p': 8}


                corresponding_moves = {}  # ходы каждой фигуры за все время
                square_figure = {}  # клетка и какая фигура на ней стоит

                # creating traking netework
                tracking_network = []
                for i in range(64):
                    tracking_network.append([0] * 64)

                # creating matrix for remembering every move of every piece
                for square in squares:
                    if board.piece_at(square):
                        string = str(board.piece_at(square)) + chess.square_name(square)
                        square_figure[chess.square_name(square)] = string
                        corresponding_moves[string] = []
                        corresponding_moves[string].append(chess.square_name(square))

                # creating matricies
                game_positional_network = []
                game_tracking_network = []
                game_attack_matrix = []
                game_features_positional=[]
                game_features_traking = []
                game_shanon = []

                for move in game.mainline_moves():

                    # print(square_figure)
                    # if move.uci()[-1].isalpha():
                    #     print(board, "\n")
                    #     print(move.uci())

                    figure_attack = {}
                    positional_network = []
                    move_attack_matrix = []
                    features_positional = []
                    features_traking = []

                    white_legal_moves = 20 # start number of legal_moves
                    black_legal_moves = 20
                    if board.turn:
                        white_legal_moves = board.legal_moves.count()
                    else:
                        black_legal_moves = board.legal_moves.count()

                    # positional netework
                    for square in squares:
                        positional_network.append([0] * 64)
                        if board.piece_at(square):
                            attacked_squares = board.attacks(square)
                            for i in attacked_squares:
                                positional_network[square][i] = 1

                    #move_attack_matrix
                    for i, figure in enumerate(corresponding_moves.keys()):
                        move_attack_matrix.append([0] * 64)
                        square = corresponding_moves[figure][-1]
                        if square!=0:
                            attacked_squares = board.attacks(chess.parse_square(square[:2]))
                            for j in attacked_squares:
                                move_attack_matrix[i][j] = 1


                    # if game.headers['Black'] == 'Shabalov, Alexander' and (move.uci() == 'g5h6'):
                    #     print(board, "\n")
                    #     print(square_figure)
                    #     print(corresponding_moves)
                    #     print(move.uci())
                    # else:
                    #     print(count)
                    #     print(game.headers)
                    #     print(board, "\n")
                    #     print(square_figure)
                    #     print(move.uci())

                    # function for updating tracking_netework
                    fool_write_tracking_netework(move, tracking_network,
                                                 square_figure)

                    # function for updating corresponding_moves matrix
                    fool_make_move_and_write_it(move, corresponding_moves,
                                                square_figure, count_pieces)

                    # positional features
                    move_singular_values = list(np.linalg.svd(move_attack_matrix)[1])[0:10]
                    features_positional = move_singular_values
                    pos_net_graph = nx.DiGraph(np.array(positional_network) > 0)

                    graph = nx.DiGraph(np.array(positional_network) > 0)


                    min_deg_more0 = 128 #max number
                    max_deg = 0
                    avg_deg = 0
                    for i in graph:
                        degree = graph.degree(i)
                        if degree < min_deg_more0:
                            min_deg_more0 = degree

                        if degree > max_deg:
                            max_deg = degree

                        avg_deg += degree
                    features_positional.append(min_deg_more0)
                    features_positional.append(max_deg)

                    avg_deg /= 64
                    features_positional.append(avg_deg)

                    density = nx.density(graph)
                    features_positional.append(density)

                    count_edges = nx.number_of_edges(graph)
                    features_positional.append(count_edges)

                    number_of_isolates = nx.number_of_isolates(graph)
                    features_positional.append(number_of_isolates)

                    sorted_len_WCC = set([len(c) for c in sorted(nx.weakly_connected_components(graph), key=len)])
                    sorted_len_WCC.discard(1)
                    if len(sorted_len_WCC) == 0:
                        sorted_len_WCC = [0]
                    else:
                        sorted_len_WCC = sorted(sorted_len_WCC)
                    sorted_len_SCC = set([len(c) for c in sorted(nx.strongly_connected_components(graph), key=len)])
                    sorted_len_SCC.discard(1)
                    if len(sorted_len_SCC) == 0:
                        sorted_len_SCC = [0]
                    else:
                        sorted_len_SCC = sorted(sorted_len_SCC)

                    count_WCC = len(sorted_len_WCC)
                    count_SCC = len(sorted_len_SCC)
                    features_positional.append(count_WCC)
                    features_positional.append(count_SCC)

                    min_WCC = sorted_len_WCC[0]
                    min_SCC = sorted_len_SCC[0]
                    features_positional.append(min_WCC)
                    features_positional.append(min_SCC)

                    max_WCC = sorted_len_WCC[-1]
                    max_SCC = sorted_len_SCC[-1]
                    features_positional.append(max_WCC)
                    features_positional.append(max_SCC)

                    summ_WCC = sum(sorted_len_WCC)
                    summ_SCC = sum(sorted_len_SCC)
                    features_positional.append(summ_WCC)
                    features_positional.append(summ_SCC)

                    avg_WCC = summ_WCC/count_WCC
                    avg_SCC = summ_SCC/count_SCC
                    features_positional.append(avg_WCC)
                    features_positional.append(avg_SCC)

                    clustering_coeficcient = nx.average_clustering(graph)
                    features_positional.append(clustering_coeficcient)

                    # features traking netework
                    move_singular_values = list(
                        np.linalg.svd(tracking_network)[1])[0:10]
                    features_traking = move_singular_values
                    pos_net_graph = nx.DiGraph(np.array(tracking_network) > 0)

                    graph = nx.DiGraph(np.array(tracking_network) > 0)

                    min_deg_more0 = 64  # max number
                    max_deg = 0
                    avg_deg = 0
                    for i in graph:
                        degree = graph.degree(i)
                        if degree < min_deg_more0:
                            min_deg_more0 = degree

                        if degree > max_deg:
                            max_deg = degree

                        avg_deg += degree
                    features_traking.append(min_deg_more0)
                    features_traking.append(max_deg)

                    avg_deg /= 64
                    features_traking.append(avg_deg)

                    density = nx.density(graph)
                    features_traking.append(density)

                    count_edges = nx.number_of_edges(graph)
                    features_traking.append(count_edges)

                    number_of_isolates = nx.number_of_isolates(graph)
                    features_traking.append(number_of_isolates)

                    sorted_len_WCC = set([len(c) for c in sorted(
                        nx.weakly_connected_components(graph), key=len)])
                    sorted_len_WCC.discard(1)
                    if len(sorted_len_WCC)==0:
                        sorted_len_WCC = [0]
                    else:
                        sorted_len_WCC = sorted(sorted_len_WCC)
                    sorted_len_SCC = set([len(c) for c in sorted(
                        nx.strongly_connected_components(graph), key=len)])
                    sorted_len_SCC.discard(1)
                    if len(sorted_len_SCC)==0:
                        sorted_len_SCC = [0]
                    else:
                        sorted_len_SCC = sorted(sorted_len_SCC)

                    count_WCC = len(sorted_len_WCC)
                    count_SCC = len(sorted_len_SCC)
                    features_traking.append(count_WCC)
                    features_traking.append(count_SCC)

                    min_WCC = sorted_len_WCC[0]
                    min_SCC = sorted_len_SCC[0]
                    features_traking.append(min_WCC)
                    features_traking.append(min_SCC)

                    max_WCC = sorted_len_WCC[-1]
                    max_SCC = sorted_len_SCC[-1]
                    features_traking.append(max_WCC)
                    features_traking.append(max_SCC)

                    summ_WCC = sum(sorted_len_WCC)
                    summ_SCC = sum(sorted_len_SCC)
                    features_traking.append(summ_WCC)
                    features_traking.append(summ_SCC)

                    avg_WCC = summ_WCC / count_WCC
                    avg_SCC = summ_SCC / count_SCC
                    features_traking.append(avg_WCC)
                    features_traking.append(avg_SCC)

                    clustering_coeficcient = nx.average_clustering(graph)
                    features_traking.append(clustering_coeficcient)

                    # count Shannon evaluation function
                    Shanon = (count_pieces['K'] - count_pieces['k'])*500 + \
                             (count_pieces['Q'] - count_pieces['q']) * 9 + \
                             (count_pieces['R'] - count_pieces['r']) * 5 + \
                             (count_pieces['N'] - count_pieces['n'] +
                             count_pieces['B'] - count_pieces['b']) * 3 + \
                             count_pieces['P'] - count_pieces['p'] + \
                             (white_legal_moves - black_legal_moves) * 0.1
                    # if Shanon > 0:
                    #     Shanon = 1
                    # elif Shanon < 0:
                    #     Shanon = -1

                    # updating all matrixes with data for this move
                    game_attack_matrix.append(move_attack_matrix)
                    game_tracking_network.append(tracking_network)
                    game_positional_network.append(positional_network)
                    game_features_positional.append(features_positional)
                    game_features_traking.append(features_traking)
                    game_shanon.append(Shanon)

                    # next_move
                    board.push(move)

                # updating
                games_attack_matrix.append(game_attack_matrix)
                games_positional_network.append(game_positional_network)
                games_tracking_network.append(game_tracking_network)
                games_features_positional.append(game_features_positional)
                games_features_traking.append(game_features_traking)
                games_results.append(result)
                games_shanon.append(game_shanon)

                # limiting on how much games we need
                count += 1
                print(count/20)

                if count==2000:
                    break
                else:
                    game = chess.pgn.read_game(pgn)

            else:
                game = chess.pgn.read_game(pgn)

    if count==2000:
        break

# print(len(games_attack_matrix), len(games_positional_network), len(games_tracking_network))

df = pd.DataFrame(data={'Result': games_results,
                        'Game_attack_matrix': games_attack_matrix,
                        'Positional_network': games_positional_network,
                        'Tracking_network': games_tracking_network,
                        'Features_positional': games_features_positional,
                        'Shanon': games_shanon,
                        'Features_traking': games_features_traking
                        })


# df = pd.read_csv('database.csv', delimiter='\t')
print(df['Result'].value_counts())
print(df.head(10))


max_moves_number = df['Features_positional'].str.len().max()
avg_moves_number = df['Features_positional'].str.len().mean()
print(max_moves_number)
print(avg_moves_number)
# df_features = df_features.replace(np.nan, 0)


columns_names = []
for i in range(max_moves_number):
    for j in range(len(df.iloc[0]['Features_positional'][0])):
        columns_names.append(f'{i}_{j}')
df_features = pd.DataFrame(columns=columns_names)
# print(df_features)
# print(len(columns_names))

for i in range(df.shape[0]):
    features_vector = []
    for j in range(len(df.iloc[i]['Features_positional'])):
        for k in range(len(df.iloc[i]['Features_positional'][0])):
            features_vector.append(df.iloc[i]['Features_positional'][j][k])
    if len(df.iloc[i]['Features_positional']) < max_moves_number:
        features_vector = features_vector[0:27]*(max_moves_number-len(df.iloc[i]['Features_positional'])) + features_vector
    df_features.loc[i] = features_vector

# print(df_features)


columns_names = []
for i in range(max_moves_number):
    for j in range(len(df.iloc[0]['Features_traking'][0])):
        columns_names.append(f'{i}_{j}')
df_features_posotional = pd.DataFrame(columns=columns_names)

for i in range(df.shape[0]):
    features_vector = []
    for j in range(len(df.iloc[i]['Features_traking'])):
        for k in range(len(df.iloc[i]['Features_traking'][0])):
            features_vector.append(df.iloc[i]['Features_traking'][j][k])
    if len(df.iloc[i]['Features_traking']) < max_moves_number:
        features_vector = features_vector[0:27]*(max_moves_number-len(df.iloc[i]['Features_traking'])) + features_vector
    df_features_posotional.loc[i] = features_vector

df_shanon = pd.DataFrame(columns=range(max_moves_number))
for i in range(df.shape[0]):
    shanons_vector = []
    for j in range(len(df.iloc[i]['Shanon'])):
        shanons_vector.append(df.iloc[i]['Shanon'][j])
    if len(df.iloc[i]['Shanon']) < max_moves_number:
        shanons_vector = ([0]*(max_moves_number-len(df.iloc[i]['Shanon']))) + shanons_vector
    df_shanon.loc[i] = shanons_vector


df_features_posotional['Result'] = np.where(df["Result"]=='1-0', True, False)
df_features['Result'] = np.where(df["Result"]=='1-0', True, False)
df_shanon['Result'] = np.where(df["Result"]=='1-0', True, False)

print(df_features_posotional)
print(df_features)
print(df_shanon)

df_features_posotional.to_csv('database_traking.csv', sep='\t', index=False)
df_features.to_csv('database_positional.csv', sep='\t', index=False)
df_shanon.to_csv('database_shanon.csv', sep='\t', index=False)

target = 'Result'

results = []
model = LogisticRegression(
    class_weight='balanced'
)


for i in range(max_moves_number-1, 1, -1):
    if i >= 7:
        df_model = pd.DataFrame(df_shanon.iloc[:, i-7:i])
        X_train_shan, X_test_shan, y_train_shan, y_test_shan = train_test_split(df_model, df_shanon[target], test_size=0.2)
        model.fit(X_train_shan, y_train_shan)
        y_predicted = model.predict(X_test_shan)
    else:
        df_model = pd.DataFrame(df_shanon.iloc[:, 0:i])
        X_train_shan, X_test_shan, y_train_shan, y_test_shan = train_test_split(
            df_model,
            df_shanon[target], test_size=0.2)
        model.fit(X_train_shan, y_train_shan)
        y_predicted = model.predict(X_test_shan)
    print(f"Test R2: {accuracy_score(y_test_shan, y_predicted)}")
    print(Counter(y_predicted))
    results.append(accuracy_score(y_test_shan, y_predicted))

plt.plot(results)
plt.show()



# G = nx.DiGraph(np.array(df.iloc[0]['Positional_network'][-1])>0)
# H = nx.relabel_nodes(G, lambda x: chess.square_name(x-1))
# print(H.degree())
# if nx.is_planar(H):
#     nx.draw_planar(H, with_labels=True)
# else:
#     nx.draw_kamada_kawai(H, with_labels=True)
# plt.title('Positional_network')
# plt.show()
#
# G = nx.DiGraph(np.array(df.iloc[0]['Tracking_network'][-1])>0)
# H = nx.relabel_nodes(G, lambda x: chess.square_name(x-1))
# print(H.degree())
# if nx.is_planar(H):
#     nx.draw_planar(H, with_labels=True)
# else:
#     nx.draw_kamada_kawai(H, with_labels=True)
# plt.title('Tracking_network')
# plt.show()