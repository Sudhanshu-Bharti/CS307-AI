import re
import heapq

# Step 2: Levenshtein Distance Calculation
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    i = 0
    while i < len(s1):
        c1 = s1[i]
        current_row = [i + 1]
        j = 0
        while j < len(s2):
            c2 = s2[j]
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
            j += 1
        previous_row = current_row
        i += 1

    return previous_row[-1]

# Step 3: A* Search Algorithm
def a_star_search(doc1_sentences, doc2_sentences):
    start_state = (0, 0, 0)  # (index in doc1, index in doc2, cumulative cost)
    goal_state = (len(doc1_sentences), len(doc2_sentences))
    open_list = []
    heapq.heappush(open_list, (0, start_state))
    came_from = {}
    cost_so_far = {start_state: 0}

    while open_list:
        _, current_state = heapq.heappop(open_list)
        i, j, g = current_state

        if (i, j) == goal_state:
            return reconstruct_path(came_from, current_state)

        neighbors = get_neighbors(current_state, doc1_sentences, doc2_sentences)
        for k in range(len(neighbors)):
            next_state, cost = neighbors[k]
            new_cost = g + cost
            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                priority = new_cost + estimate_heuristic(next_state, doc1_sentences, doc2_sentences)
                heapq.heappush(open_list, (priority, next_state))
                came_from[next_state] = current_state

    return []  # Return an empty path if no solution is found


# Step 1: Text Preprocessing
def preprocess_text(text):
    sentences = re.split(r'(?<=[.!?]) +', text.lower())
    sentences = [re.sub(r'[^\w\s]', '', sentence) for sentence in sentences]
    return sentences

# Step 4: Detecting Plagiarism
def detect_plagiarism(doc1, doc2, threshold=5):
    doc1_sentences = preprocess_text(doc1)
    doc2_sentences = preprocess_text(doc2)

    alignment = a_star_search(doc1_sentences, doc2_sentences)

    potential_plagiarism = []
    for (i, j, g) in alignment:
        if i > 0 and j > 0:
            distance = levenshtein_distance(doc1_sentences[i-1], doc2_sentences[j-1])
            print(f"Checking : Doc1 : '{doc1_sentences[i-1]}' \n  \t\t Doc2 : '{doc2_sentences[j-1]}' \n Distance: {distance} \n\n")
            if distance <= threshold:  # Set an appropriate threshold value
                potential_plagiarism.append((doc1_sentences[i-1], doc2_sentences[j-1], distance))

    print("\nFound Potential Plagiarism:", potential_plagiarism)
    return potential_plagiarism


def get_neighbors(state, doc1_sentences, doc2_sentences):
    i, j, g = state
    neighbors = []

    if i < len(doc1_sentences) and j < len(doc2_sentences):
        cost = levenshtein_distance(doc1_sentences[i], doc2_sentences[j])
        neighbors.append(((i + 1, j + 1, g + cost), cost))

    return neighbors

def estimate_heuristic(state, doc1_sentences, doc2_sentences):
    i, j, _ = state
    remaining_doc1 = len(doc1_sentences) - i
    remaining_doc2 = len(doc2_sentences) - j

    # Heuristic based on the remaining number of sentences
    return abs(remaining_doc1 - remaining_doc2)

def reconstruct_path(came_from, current_state):
    path = []
    while current_state in came_from:
        path.append(current_state)
        current_state = came_from[current_state]
    path.reverse()
    return path


# Example Usage
doc1 = "My Pasta is warm. Water is blue. She is so smart."
doc2 = "My Pasta is cold. Water is so blue. She is so smart."

plagiarized_pairs = detect_plagiarism(doc1, doc2)

# Output the detected plagiarism pairs
print("\nDetected Plagiarism Pairs:")
print("================================")
for pair in plagiarized_pairs:
    print(f"Doc1: '{pair[0]}'\nDoc2: '{pair[1]}'\nFounded Distance: {pair[2]}\n")
    print("================================")