from sklearn import tree

# 0= Rock, 1 = Paper, 2 = Scissors
x = [[0, 1], [1, 2], [2, 0]]
y = [1, 2, 0]
clf = tree.DecisionTreeClassifier()
clf.fit(x, y)
next_move = [[0, 1]]
prediction = clf.predict(next_move)
print(f"JARVIS predicts you'll pick steps {next_move}: {prediction[0]}");