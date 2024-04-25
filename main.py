import nltk
import heapq

def get_text():
    text = """If you've tried cannabis you may have experienced some of its side effects, perhaps you got a bit peckish and you've got the local pizza shop on speed dial or maybe you got a bit sleepy. So, isn't it a no-brainer that cannabis can help us sleep better? Well, I've been researching this topic as a sleep physiologist and I'm here to talk about why cannabis for treating insomnia is complicated.

    Humans have been using cannabis for at least 5,000 years. We've been using it to make clothes, for building materials, it's been used in religious ceremonies and for its health benefits, and apparently, it's been used for fun. But it's only been in the past 100 or so years that we've started understanding the science of cannabis and its effects.
    
    We know that the cannabis plant contains hundreds of chemical compounds, including the cannabinoids. The most well-known cannabinoid is Delta 9 Tetrahydrocannabinol, you may have heard this as THC. THC is primarily known for its intoxicating properties, that high that people tend to seek out when they use cannabis recreationally and when you're pulled over for those roadside drug tests, it's the THC that they're looking for.
    
    The other well-known cannabinoid is cannabidiol or CBD. Unlike THC, CBD is non-intoxicating and as recently as about 30 years ago, scientists also discovered other cannabinoids, ones that are produced by our own bodies, the endocannabinoids. It's the endocannabinoids that are produced in our brain and throughout our body that are thought to cause that high that people experience after running or that relaxed post-exercise feeling. Don't worry, these ones are not being detected by those roadside drug tests so you can continue with your marathon training.
    
    But what's really exciting about this rapid increase in our understanding and knowledge of cannabis and cannabinoids is that the cannabis plant or cannabis was illegal in most countries around the world until the late 1990s. So it's only been in the past 20 or so years since it's been legalized, at least for medicinal use in some countries, that its use has been increasing and we're understanding more about its benefits.
    
    There's now pretty solid evidence that cannabinoids can help treat rare types of epilepsy, the nausea and vomiting associated with some cancer treatments. It can also help treat some forms of chronic or long-term pain, the muscle stiffness associated with multiple sclerosis, and in patients with HIV/AIDS, it can improve appetite. And there is also some evidence that cannabinoids may be helpful in reducing anxiety associated with public speaking. I didn't make it up and I haven't had any medicinal preparation today.
    
    Cannabinoids might also be helpful for treating some sleep disorders. Most commonly, it's been used to treat insomnia. In fact, some surveys report that up to 47% of people who use cannabis medicinally are using it to improve their sleep. Insomnia is the most prevalent of the sleep disorders. It affects a third of us and for 15% to us, it is a chronic problem so it lasts longer than 3 months. The symptoms of insomnia are variable. You may have trouble falling asleep, you may have trouble staying asleep and if you're really unlucky you might experience both. But even if you haven't experienced insomnia yourself, you can probably relate to the feelings of not having had enough sleep and how it impacts you the next day. Perhaps you're not as patient with your loved ones, you might find it hard to concentrate or stay alert or you might find it difficult to remember things like finding the right words when you're doing a public talk that you've had to memorize. In the long term, insomnia can contribute to conditions like anxiety and depression as well as some forms of cardiovascular disease. But we have a good treatment for insomnia, cognitive behavioral therapy for insomnia or CBT-I which is typically done under the guidance of a specialist sleep psychologist. But it can take weeks to see benefits from CBT-I and it can be difficult to access. So wouldn't it be great if we had an alternative treatment for insomnia that was safe, easy to access, and gave us quick results?
    
    Well, we know that cannabis has been used to help sleep, probably for thousands of years and there's plenty of reports of improved sleep in people who have used cannabinoids for treating other medical conditions. We just don't have good evidence that it can help with insomnia. So our team at the center of asleep science at the University of Western Australia in Perth recently teamed with ziler therapeutics to investigate the effects of a cannabinoid medication on chronic insomnia. In this world-first study, 24 participants took a cannabinoid medication which contained THC and CBD as well as another cannabinoid, cannabiol or CBN, for 2 weeks. They also took a placebo for 2 weeks in random order. They didn't know which one they were having and we didn't either until we'd analyzed all of the data.
    
    Over the two weeks, we measured their sleep with a wristwatch-type device like a research-quality smartwatch and we also made more sophisticated measures of their sleep over a single night while they slept in a sleep laboratory. We found that when people took the cannabinoid medication, they actually didn't sleep much better when they were in the laboratory. This may be because it was just a single night or it may be because they had to sleep with equipment like this. What we did find was that when people were sleeping at home for 2 weeks, as we measured their sleep with that fancy wristwatch, that they slept on average 33 minutes a night longer and they were awake for 10 minutes less each night. They also reported that they felt they slept better and they felt more rested on waking up each day. No one reported an increase in pizza consumption. Seriously though, this is the most comprehensive investigation of any medicinal cannabis product as a treatment for insomnia that's ever been done and the results are really exciting.
    
    So, does it mean that if you've got insomnia you should be sitting back on the couch each night with a joint? Well, this is where things get complicated. Firstly, cannabis remains illegal in most countries around the world so please don't do that. Secondly, like smoking cigarettes, smoking cannabis is associated with negative long-term health consequences so it's not recommended. There are far safer ways that it can be consumed. Also, if you consume cannabis that you just got from a friend, you won't know exactly what's in it. With medicinal cannabis, we know the concentrations of each of the cannabinoids so we can work out exactly how much of each cannabinoid you're having. But even though we might know how much of each of the cannabinoids you're having, we also know that like any medication, your response to it might be variable. So, what might make me happy and docile might make you energetic and want to clean the house. And just saying, if that happens, feel free to come over to my place.
    
    But one of the main reasons that the jury is still out on whether we should be using cannabinoids to treat insomnia is that although the results from our study were really encouraging, it's just one study using one combination of cannabinoids and we studied 24 people who were extensively screened to have no other major health condition and to be pretty much taking no other medication. We really need more research in a larger, more diverse group of people using different formulations and combinations of cannabinoids to really be convinced of its benefit and safety.
    
    You still can consume too much and it can have negative effects on your physical and mental health and plenty of people have done really silly or dangerous things after having cannabis. However, a lot of our understanding of these risks associated with using cannabis have come from studying people who use it recreationally. Again, we need more evidence looking at the effects of using cannabis or cannabinoids in the doses and the populations that use it medically.
    
    As you can see, answering the question about whether we should use medicinal cannabis to treat sleep disorders isn't so simple. To have an informed conversation about the usefulness of cannabinoids for treating medical conditions, including insomnia, we really need to generate more evidence and to understand the science better. Our research here in Perth is leading the way in answering some of these complicated questions and the early data is really promising. But the data party has only just started.
    
    For those of you who were hoping to get the green light to use cannabis to get a better night's sleep, you'll just need to wait for the evidence to grow. """

    return text

def token(text):
    word_list = nltk.word_tokenize(text)
    return word_list

def sent_token(text):
    sent_list = nltk.sent_tokenize(text)
    return sent_list

def remove_stopwords(word_list):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    filtered_words = []

    for word in word_list:
        if word not in stopwords:
            filtered_words.append(word)

    return filtered_words

def word_freq(filtered_words):
    word_frequency = {}

    for word in filtered_words:
        if word not in word_frequency:
            word_frequency[word] = 1
        else:
            word_frequency[word] += 1

    return word_frequency

def max_freq(word_frequency):
    max_freq = max(word_frequency.values())

    for word in word_frequency.keys():
        word_frequency[word] = (word_frequency[word]/max_freq)

    return word_frequency

def sentence_scores(sent_list, word_frequency):
    sent_scores = {}

    for sentence in sent_list:
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word_frequency.keys():
                if len(sentence.split(' ')) < 30:
                    if sentence not in sent_scores.keys():
                        sent_scores[sentence] = word_frequency[word]
                    else:
                        sent_scores[sentence] += word_frequency[word]

    return sent_scores

def get_summary(sent_scores):
    summary = heapq.nlargest(7, sent_scores, key=sent_scores.get)
    return summary

def main():
    text = get_text()
    word_list = token(text)
    sent_list = sent_token(text)
    filtered_words = remove_stopwords(word_list)
    word_frequency = word_freq(filtered_words)
    word_frequency = max_freq(word_frequency)
    sent_scores = sentence_scores(sent_list, word_frequency)
    summary = get_summary(sent_scores)

    for a in summary:
        print(a)
        print('\n')


if __name__ == '__main__':
    main()


