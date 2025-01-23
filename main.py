from transformers import pipeline


TEXT = """ The Battle of Alesia or siege of Alesia (September 52 BC) was the climactic military engagement of the Gallic Wars, fought around the Gallic oppidum (fortified settlement) of Alesia in modern France, a major centre of the Mandubii tribe. It was fought by the Roman army of Julius Caesar against a confederation of Gallic tribes united under the leadership of Vercingetorix of the Arverni. It was the last major engagement between Gauls and Romans, and is considered one of Caesar's greatest military achievements and a classic example of siege warfare and investment; the Roman army built dual lines of fortifications—an inner wall to keep the besieged Gauls in, and an outer wall to keep the Gallic relief force out. The Battle of Alesia marked the end of Gallic independence in the modern day territory of France and Belgium.

The battle site was probably atop Mont Auxois, above modern Alise-Sainte-Reine in France, but this location, some have argued, does not fit Caesar's description of the battle. A number of alternatives have been proposed over time, among which only Chaux-des-Crotenay (in Jura in modern France) remains a challenger today.[10]

The event is described by Caesar himself in his Commentarii de Bello Gallico as well as several later ancient authors (namely Plutarch and Cassius Dio). After the Roman victory, Gaul (very roughly modern France) was subdued, although Gallic territories north of Gallia Narbonensis would not become a Roman province until 27 BC. The Roman Senate granted Caesar a thanksgiving of 20 days for his victory in the Gallic War.[11]
"""


#MODEL 1
summarizerModel1 = pipeline("summarization", model="facebook/bart-large-cnn")

print(summarizerModel1(TEXT, max_length=130, min_length=30, do_sample=False))


#MODEL 2
summarizerModel2 = pipeline("summarization", model="Falconsai/text_summarization")

print(summarizerModel2(TEXT, max_length=130, min_length=30, do_sample=False))

#MODEL 3

summarizerModel3 = pipeline("summarization", model="pszemraj/led-large-book-summary")

print(summarizerModel3(TEXT, max_length=130, min_length=30, do_sample=False))

#MODEL 4


summarizerModel4 = pipeline("summarization", model='it5/it5-base-news-summarization')

print(summarizerModel4(TEXT, max_length=130, min_length=30, do_sample=False))



