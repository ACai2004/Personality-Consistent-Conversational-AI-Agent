import nlpaug.augmenter.word as naw

aug = naw.SpellingAug()

text = "I'm gonna fly up because I can fly."
print(aug.augment(text, n=3))