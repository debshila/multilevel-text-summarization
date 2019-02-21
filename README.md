# Multilevel Text Summarization
## Insight S19 Project
The goal of this project was to develop a way to make information of an unfamiliar knowledge domain ingestible for a naive learner. The project leverages Topic modeling using an LDA to identify underlying themes within documents in a specific domain, followed by making gists within each domain using a variation of the TextRank algorithm as implemented in the Gensim package. The assumption is that the topics would help the learner naviagate the domain, and give the learners the opportunity to prioritize information in the domain by their representation in the knowledge corpus. Additionally, the summaries will highlight the most important content within each topic.

For the purposes of this project, I focused on understanding the latent themes in privacy policy documents. The selection of this domain was motivated by the recent interest in privacy policy documentation for various services after various regulatory changes (e.g. GDPR). The documents used were obtained from the Usable Privacy Project (ACL/COLING dataset retrieved from https://usableprivacy.org/data) 

Try my web app to examine the most critical sentences from privacy policy documents while also observing common themes underlying similar privacy policy documents: https://gist-do-it.herokuapp.com/


