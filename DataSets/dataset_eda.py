import pandas as pd


def unicode_descr_to_int(uc):
  # Extract the code point (removing 'U+')
  code_point = int(uc[2:], 16)
  # Convert the code point to the character
  character = chr(code_point)
  return ord(character)


def get_kuzushiji_labels(annotations_file):
    df = pd.read_csv(annotations_file)
    labels = set()
    for _, row in df.iterrows():
      label_string = row["labels"]
      tokens = label_string.split()
      for i in range(0, len(tokens), 5):
          unicode_char = tokens[i]
          labels.add(unicode_char)
    return labels
	
	
def get_kuzushiji_stats(annotations_file):
    df = pd.read_csv(annotations_file)
    stats = {}
    for _, row in df.iterrows():
      label_string = row["labels"]
      tokens = label_string.split()
      for i in range(0, len(tokens), 5):
          unicode_char = tokens[i]
          if unicode_char in stats:
            stats[unicode_char] +=1
          else :
            stats[unicode_char] = 1
      sorted_stats = sorted(stats.items(), key=lambda item: item[1], reverse=True)
    return sorted_stats
    
def unicode_labels_to_ints(annotations_file):
    unicode_labels = get_kuzushiji_labels(annotations_file)
    labels_dict = {}
    for ul in unicode_labels:
      labels_dict[ul] = unicode_descr_to_int(ul)
    sorted_items = sorted(labels_dict.items(), key=lambda item: item[1])
    ordered_labels = {key: idx for idx, (key, _) in enumerate(sorted_items)}
    return ordered_labels

def convert_labels(labels_stats):
    labels_dict = {}
    for ul in labels_stats:
      labels_dict[ul] = unicode_descr_to_int(ul)
    sorted_items = sorted(labels_dict.items(), key=lambda item: item[1])
    ordered_labels = {key: idx for idx, (key, _) in enumerate(sorted_items)}
    return ordered_labels