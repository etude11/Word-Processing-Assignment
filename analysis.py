import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import nltk
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

rt_data = pd.read_csv('dataset/processed_RTs.tsv', sep='\t')
print(f"RT data shape: {rt_data.shape}")

mean_rt = rt_data.groupby('word')['RT'].mean().reset_index()
mean_rt.columns = ['word', 'mean_RT']
mean_rt['length'] = mean_rt['word'].str.len()

freqs = []
for order in [1, 2, 3, 4]:
    freq_df = pd.read_csv(f'dataset/freqs/freqs-{order}.tsv', sep='\t', header=None, 
                          names=['token_code', 'ngram_order', 'token', 'ngram_freq', 'prev_freq'])
    freq_df['order'] = order
    freqs.append(freq_df)
freqs_all = pd.concat(freqs)

unigram_freq = freqs_all[freqs_all['order'] == 1].copy()
unigram_freq = unigram_freq[['token', 'ngram_freq']].drop_duplicates()
unigram_freq.columns = ['word', 'word_freq']

trigram_freq = freqs_all[freqs_all['order'] == 3].copy()
trigram_freq['cond_prob'] = trigram_freq['ngram_freq'] / trigram_freq['prev_freq']
trigram_freq['neg_log_prob'] = -np.log(trigram_freq['cond_prob'] + 1e-10)
trigram_freq = trigram_freq[['token', 'neg_log_prob']].drop_duplicates()
trigram_freq.columns = ['word', 'neg_log_prob']

data = mean_rt.merge(unigram_freq, on='word', how='left')
data = data.merge(trigram_freq, on='word', how='left')
data = data.dropna()

print("\n=== PART I: PRELIMINARY DATA ANALYSIS ===\n")

print(f"Mean RT per word computed: {len(data)} unique words")
print(f"Average mean RT: {data['mean_RT'].mean():.2f} ms")

plt.figure()
data_sample = data.sample(min(5000, len(data)))
plt.scatter(data_sample['length'], data_sample['mean_RT'], alpha=0.3, s=10)
plt.xlabel('Word Length (characters)')
plt.ylabel('Mean RT (ms)')
plt.title('Word Length vs Mean Reading Time')
plt.savefig('plot_length_rt.png', dpi=300, bbox_inches='tight')
print("Saved: plot_length_rt.png")

plt.figure()
data_freq = data[data['word_freq'] > 0].copy()
data_freq['log_freq'] = np.log(data_freq['word_freq'])
data_freq_sample = data_freq.sample(min(5000, len(data_freq)))
plt.scatter(data_freq_sample['log_freq'], data_freq_sample['mean_RT'], alpha=0.3, s=10)
plt.xlabel('Log Word Frequency')
plt.ylabel('Mean RT (ms)')
plt.title('Word Frequency vs Mean Reading Time')
plt.savefig('plot_freq_rt.png', dpi=300, bbox_inches='tight')
print("Saved: plot_freq_rt.png")

r_len_freq, p_len_freq = pearsonr(data_freq['length'], data_freq['log_freq'])
print(f"\nPearson correlation (length vs log frequency): r={r_len_freq:.4f}, p={p_len_freq:.4e}")

r_len_rt, p_len_rt = pearsonr(data['length'], data['mean_RT'])
print(f"Pearson correlation (length vs mean RT): r={r_len_rt:.4f}, p={p_len_rt:.4e}")

r_freq_rt, p_freq_rt = pearsonr(data_freq['log_freq'], data_freq['mean_RT'])
print(f"Pearson correlation (log frequency vs mean RT): r={r_freq_rt:.4f}, p={p_freq_rt:.4e}")

print("\n=== PART II: HYPOTHESIS TESTING ===\n")

data_clean = data[data['neg_log_prob'] < 100].copy()

X1 = data_clean[['word_freq', 'length']].values
X2 = data_clean[['neg_log_prob', 'length']].values
y = data_clean['mean_RT'].values

model1 = LinearRegression().fit(X1, y)
y_pred1 = model1.predict(X1)
r2_1 = r2_score(y, y_pred1)
mse_1 = mean_squared_error(y, y_pred1)

model2 = LinearRegression().fit(X2, y)
y_pred2 = model2.predict(X2)
r2_2 = r2_score(y, y_pred2)
mse_2 = mean_squared_error(y, y_pred2)

print("Hypothesis 1: LM probabilities vs word frequency")
print(f"Model 1 (freq + length): R²={r2_1:.4f}, MSE={mse_1:.2f}")
print(f"Model 2 (-log(prob) + length): R²={r2_2:.4f}, MSE={mse_2:.2f}")
print(f"Better model: Model {'2 (LM probabilities)' if r2_2 > r2_1 else '1 (word frequency)'}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_pred1, y, alpha=0.3, s=10)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Predicted RT')
plt.ylabel('Actual RT')
plt.title(f'Model 1: Freq + Length (R²={r2_1:.4f})')

plt.subplot(1, 2, 2)
plt.scatter(y_pred2, y, alpha=0.3, s=10)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Predicted RT')
plt.ylabel('Actual RT')
plt.title(f'Model 2: -log(prob) + Length (R²={r2_2:.4f})')
plt.tight_layout()
plt.savefig('hypothesis1_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: hypothesis1_comparison.png")

nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('universal_tagset', quiet=True)

words_list = data_clean['word'].tolist()
pos_tags = nltk.pos_tag(words_list, tagset='universal')
pos_dict = {word: pos for word, pos in pos_tags}
data_clean['pos'] = data_clean['word'].map(pos_dict)

content_pos = ['NOUN', 'VERB', 'ADJ', 'ADV']
function_pos = ['ADP', 'DET', 'CONJ', 'PRON', 'PRT']

data_content = data_clean[data_clean['pos'].isin(content_pos)].copy()
data_function = data_clean[data_clean['pos'].isin(function_pos)].copy()

print(f"\nHypothesis 2: Content vs Function words")
print(f"Content words: {len(data_content)}, Function words: {len(data_function)}")

if len(data_content) > 10 and len(data_function) > 10:
    Xc1 = data_content[['word_freq', 'length']].values
    Xc2 = data_content[['neg_log_prob', 'length']].values
    yc = data_content['mean_RT'].values
    
    modelc1 = LinearRegression().fit(Xc1, yc)
    r2_c1 = r2_score(yc, modelc1.predict(Xc1))
    
    modelc2 = LinearRegression().fit(Xc2, yc)
    r2_c2 = r2_score(yc, modelc2.predict(Xc2))
    
    Xf1 = data_function[['word_freq', 'length']].values
    Xf2 = data_function[['neg_log_prob', 'length']].values
    yf = data_function['mean_RT'].values
    
    modelf1 = LinearRegression().fit(Xf1, yf)
    r2_f1 = r2_score(yf, modelf1.predict(Xf1))
    
    modelf2 = LinearRegression().fit(Xf2, yf)
    r2_f2 = r2_score(yf, modelf2.predict(Xf2))
    
    print(f"Content words - Model 1 (freq): R²={r2_c1:.4f}")
    print(f"Content words - Model 2 (-log prob): R²={r2_c2:.4f}")
    print(f"Function words - Model 3 (freq): R²={r2_f1:.4f}")
    print(f"Function words - Model 4 (-log prob): R²={r2_f2:.4f}")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(['Freq', '-log(prob)'], [r2_c1, r2_c2])
    plt.ylabel('R²')
    plt.title('Content Words')
    plt.ylim([0, max(r2_c1, r2_c2) * 1.2])
    
    plt.subplot(1, 2, 2)
    plt.bar(['Freq', '-log(prob)'], [r2_f1, r2_f2])
    plt.ylabel('R²')
    plt.title('Function Words')
    plt.ylim([0, max(r2_f1, r2_f2) * 1.2])
    plt.tight_layout()
    plt.savefig('hypothesis2_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: hypothesis2_comparison.png")

print("\n=== PART III: FOBS MODEL ===\n")

nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()

data_clean['lemma'] = data_clean['word'].apply(lambda x: lemmatizer.lemmatize(x.lower()))
data_clean['lemma_length'] = data_clean['lemma'].str.len()

lemma_freqs = data_clean.groupby('lemma')['word_freq'].sum().reset_index()
lemma_freqs.columns = ['lemma', 'lemma_freq']
data_clean = data_clean.merge(lemma_freqs, on='lemma', how='left')

print("Hypothesis 1: Root frequency vs surface frequency")

data_lemma = data_clean[data_clean['lemma_freq'] > 0].copy()

Xl1 = data_lemma[['word_freq', 'length']].values
Xl2 = data_lemma[['lemma_freq', 'lemma_length']].values
yl = data_lemma['mean_RT'].values

modell1 = LinearRegression().fit(Xl1, yl)
r2_l1 = r2_score(yl, modell1.predict(Xl1))

modell2 = LinearRegression().fit(Xl2, yl)
r2_l2 = r2_score(yl, modell2.predict(Xl2))

print(f"Model 1 (surface freq + length): R²={r2_l1:.4f}")
print(f"Model 2 (lemma freq + length): R²={r2_l2:.4f}")
print(f"Better model: Model {'2 (lemma frequency)' if r2_l2 > r2_l1 else '1 (surface frequency)'}")

print("\nHypothesis 2: Pseudo-affixed vs regular affixed words")

pseudo_affixed = ['finger', 'corner', 'butter', 'winter', 'number']
real_affixed = ['singer', 'owner', 'better', 'winner', 'hunter']

test_words = pseudo_affixed + real_affixed
test_data = data_clean[data_clean['word'].isin(test_words)].copy()
test_data['type'] = test_data['word'].apply(lambda x: 'pseudo' if x in pseudo_affixed else 'real')

if len(test_data) > 0:
    pseudo_rt = test_data[test_data['type'] == 'pseudo']['mean_RT'].mean()
    real_rt = test_data[test_data['type'] == 'real']['mean_RT'].mean()
    
    print(f"Pseudo-affixed words mean RT: {pseudo_rt:.2f} ms")
    print(f"Real affixed words mean RT: {real_rt:.2f} ms")
    print(f"Difference: {pseudo_rt - real_rt:.2f} ms")
    
    plt.figure()
    test_data_plot = test_data.sort_values('type')
    plt.bar(test_data_plot['word'], test_data_plot['mean_RT'], 
            color=['red' if t == 'pseudo' else 'blue' for t in test_data_plot['type']])
    plt.xticks(rotation=45)
    plt.ylabel('Mean RT (ms)')
    plt.title('Pseudo-affixed vs Real Affixed Words')
    plt.legend(['Pseudo-affixed', 'Real-affixed'])
    plt.tight_layout()
    plt.savefig('pseudo_vs_real_affixed.png', dpi=300, bbox_inches='tight')
    print("Saved: pseudo_vs_real_affixed.png")

print("\nAnalysis complete!")
