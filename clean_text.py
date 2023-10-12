import emoji
import re 


#Adds a full stop if there is not already a punctuation mark after the end of the first line
def punctuate_headline(text): 
    #find first newline 
    position = text.find('\n')
    if position > 0: 
        if text[position-1] not in ['.','?','!']:
            text = re.sub('\n', '.\n', text, count = 1)


    return text 

def clean_text(text):
    #generic 

    #remove hyperlinks 
    text = re.sub('(http|https|ftp):\/\/(\S*)', "", text) 

    #remove twitter pic urls
    text = re.sub('pic[.]twitter[.]com(\S*)', '', text) 

    #remove twitter handles in brackets such as Emily (@emily)
    text = re.sub("[(]@.*?[)]", "", text)

    #remove emails: 
    text = re.sub('([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', "", text)
    #after removing emails, remove all @s (twitter handles)
    text = re.sub('@', "", text)

    #remove hashtags
    text = re.sub("#", "", text)


    #remove all emoji
    text = emoji.replace_emoji(text, replace='')

    return text 

def clean_text_specific(text):
    text = punctuate_headline(text)

    #simple one-off replacements
    phrases_to_delete = [
        'Follow Ben Kew on Facebook, Twitter at @ben_kew, or email him at bkew@breitbart.com.',
        'Older articles by Judi McLeod',
        'Only YOU can save CFP from Social Media Suppression.',
        'Tweet, Post, Forward, Subscribe or Bookmark us',
        'Copyright © Canada Free Press',
        '© REUTERS / Toby Melville',
        '© Reuters',
        '©2018 The Atlanta Journal-Constitution (Atlanta, Ga.), Distributed by Tribune Content Agency, LLC.',
        '©2018 Los Angeles Times, Distributed by Tribune Content Agency, LLC.',
        '© EPA',
        '© AFP',
        '© Claudio Reyes/AFP/Getty Images',
        '©2018 Deutsche Presse-Agentur GmbH (Hamburg, Germany), Distributed by Tribune Content Agency, LLC.',
        '©2018 CQ-Roll Call, Inc., All Rights Reserved, Distributed by Tribune Content Agency, LLC.',
        '© Getty',
        '© Provided by Quartz',
        '© Provided by Oath Inc.',
        '© AP Photo',
        '© AFP 2018 /',
        '© Provided by Al Jazeera',
        'RSS Feed for Judi McLeod',
        'Follow him on Twitter here. Like him on Facebook here.',
        'Follow Pam Key on Twitter @pamkeyNEN',
        'Think your friends would be interested?',
        'Share this story!',
        'Article posted with permission from Daisy Luther',
        'Reprinted with permission from Consortiumnews.com.',
        'Write to Abby Vesoulis at abby.vesoulis@time.com.',
        'Contact us at editors@time.com.',
        '[emphasis added]',
        '• Email: ghamilton@nationalpost.com | Twitter: grayhamilton',
        'COPYRIGHT 2018 CREATORS.COM',
        'Patrick J. Buchanan is the author of “The Greatest Comeback: How Richard Nixon Rose From Defeat to Create the New Majority.',
        'His latest book, published May 9, is', 
        '“Nixon’s White House Wars: The Battles That Made and Broke a President and Divided America Forever.”',
        '©2018 The Virginian-Pilot (Norfolk, Va.)',
        'Visit The Virginian-Pilot at pilotonline.com',
        'Distributed by Tribune Content Agency, LLC.',
        'Amazon.com $50 Gift Ca...',
        'Check Amazon for Pricing.',
        'Copyright 2019 The Associated Press.',
        'All rights reserved.',
        'This material may not be published, broadcast, rewritten or redistributed.',
        'Jeremy C. Fox can be reached at jeremy.fox@globe.com',
        'Edwin S. Rubenstein (email him) is President of ESR Research Economic Consultants',
        '— John T. Bennett',
        'CQ-Roll Call',
        'Robert Henderson [Email him] is a retired civil servant living in London and consequently old enough to remember what life was like before political correctness.',
        'He runs the Living In A Madhouse and England Calling blogs.',
        'Contributing to this report were Mike Catalini in Morrisville, Pennsylvania; Ben Finley in Norfolk, Virginia; Claire Galofaro in Louisville, Kentucky; Allen Breed in Wake Forest, North Carolina; Jennifer McDermott in Providence, Rhode Island; Adam Beam in Frankfort, Kentucky; Lindsay Whitehurst in Salt Lake City, Utah; Hannah Grabenstein in Little Rock, Arkansas; Jeffrey Collins in Columbia, South Carolina; Roxana Hegeman in Wichita, Kansas; Sarah Blake Morgan in Charlotte, North Carolina; and Elliott Spagat in Solana Beach, California.',
        'Sedensky can be reached at msedensky@ap.org and https://twitter.com/sedensky',
        'CLICK HERE FOR THE FOX NEWS APP',
        'CLICK HERE TO GET THE FOX NEWS APP',
        'Click here to listen to this interview.',
        'WARNING: GRAPHIC LANGUAGE: Click here to watch the video.',
        'Read more from Yahoo News:',
        'Photo credit: Keilana, Roberta Mura',
        'Celestron Sky Maps Celstron Best Price: $15.97 Buy New $18.95 (as of 04:55 EDT - Details)',
        'Secret Empires: How th... Peter Schweizer Best Price: $5.19 Buy New $8.24 (as of 12:00 EDT - Details)',
        'The BBC writes: Against the State: An ... Llewellyn H. Rockwell Jr. Best Price: $12.99 Buy New $9.05 (as of 10:25 EDT - Details)',
        'Battlefield America: T... John W. Whitehead Best Price: $10.78 Buy New $19.06 (as of 09:30 EDT - Details)',
        'The WikiLeaks Files: T... WikiLeaks Best Price: $2.50 Buy New $7.00 (as of 05:50 EDT - Details)',
        'American Raj: Liberati... Eric Margolis Best Price: $5.99 Buy New $45.86 (as of 09:30 EDT - Details)',
        'War at the Top of the ... Eric Margolis Best Price: $2.60 Buy New $24.57 (as of 09:00 EDT - Details)',
        'The Tyranny of Good In... Paul Craig Roberts, La... Best Price: $5.64 Buy New $5.00 (as of 10:35 EDT - Details)',
        'No Place to Hide: Edwa... Glenn Greenwald Best Price: $1.49 Buy New $3.70 (as of 06:30 EDT - Details)',
        'The Neoconservative Th... Dr. Paul Craig Roberts Best Price: $11.96 Buy New $19.34 (as of 10:15 EDT - Details)',
        'The Russia Hoax: The I... Gregg Jarrett Best Price: $13.99 Buy New $12.63 (as of 08:40 EDT - Details)',
        'Death of a Nation: Pla... Dinesh D\'Souza Best Price: $14.99 Buy New $16.99 (as of 03:35 EDT - Details)',
        'With Liberty and Justi... Glenn Greenwald Best Price: $2.66 Buy New $11.34 (as of 02:05 EST - Details)',
        'Against the State: An ... Llewellyn H. Rockwell Jr. Best Price: $9.94 Buy New $9.95 (as of 01:45 EST - Details)',
        'The Kennedy Autopsy Jacob G Hornberger Best Price: $9.85 Buy New $9.95 (as of 03:35 EDT - Details)',
        'The Tyranny of Good In... Paul Craig Roberts, La... Best Price: $5.64 Buy New $5.00 (as of 10:35 EDT - Details)',
        'The Neoconservative Th... Dr. Paul Craig Roberts Best Price: $11.96 Buy New $19.34 (as of 02:05 EDT - Details)',
        'How America Was Lost: ... Dr. Paul Craig Roberts Best Price: $8.00 Buy New $8.51 (as of 02:00 EDT - Details)',
        'Against the State: An ... Llewellyn H. Rockwell Jr. Best Price: $9.95 Buy New $9.95 (as of 06:00 EDT - Details)',
        'The WikiLeaks Files: T... WikiLeaks Best Price: $2.50 Buy New $7.00 (as of 05:50 EDT - Details)',
        'American Raj: Liberati... Eric Margolis Best Price: $5.99 Buy New $46.29 (as of 07:40 EDT - Details)',
        'War at the Top of the ... Eric Margolis Best Price: $2.60 Buy New $24.57 (as of 06:55 EDT - Details)',
        'Nixonu2019s White Hous... Patrick J. Buchanan Best Price: $6.38 Buy New $10.01 (as of 10:15 EDT - Details)',
        'The Greatest Comeback:... Patrick J. Buchanan Best Price: $3.13 Buy New $6.00 (as of 08:50 EDT - Details)',
        'Churchill, Hitler, and... Patrick J. Buchanan Best Price: $6.20 Buy New $6.00 (as of 12:40 EDT - Details)',
        'The Deep State: The Fa... Mike Lofgren Best Price: $7.55 Buy New $6.00 (as of 11:45 EDT - Details)',
        'Special Ed: Voices fro... Dennis Bernstein Best Price: $2.49 Buy New $5.00 (as of 06:30 EDT - Details)',
        'Cypherpunks: Freedom a... Julian Assange Best Price: $3.04 Buy New $9.55 (as of 06:45 EDT - Details)',
        'The WikiLeaks Files: T... WikiLeaks Best Price: $2.75 Buy New $7.00 (as of 12:00 EDT - Details)',
        'Secret Empires: How th... Peter Schweizer Best Price: $5.19 Buy New $8.24 (as of 12:00 EDT - Details)',
        'Ship of Fools: How a S... Tucker Carlson Best Price: $2.89 Buy New $7.30 (as of 09:15 EDT - Details)',
        'Ship of Fools: How a S... Tucker Carlson Best Price: $2.89 Buy New $7.30 (as of 09:15 EDT - Details)',
        'Against the State: An ... Llewellyn H. Rockwell Jr. Best Price: $12.99 Buy New $9.19 (as of 04:05 EDT - Details)',
        'Secret Empires: How th... Peter Schweizer Best Price: $5.19 Buy New $8.24 (as of 12:00 EDT - Details)',
        'Architects of Ruin: Ho... Peter Schweizer Best Price: $2.90 Buy New $6.54 (as of 11:35 EDT - Details)',
        'Celestron NexStar 90SL... Buy New $309.95 (as of 04:50 EDT - Details)',
        'Speaking Freely: Ray M... Buy New $1.99 (as of 09:20 EDT - Details)',
        'Read the Whole Article',
        '(Read the whole story HERE.)',
        '© Claudio Reyes/AFP/Getty Images',
        'Dibyangshu Sarkar/AFP/Getty Images',
        'RICARDO ARDUENGO/AFP/Getty Images',
        '(Photo by Jack Taylor/Getty Images)',
        '[Editors\' postscript: Please make sure to FOLLOW Jamie Glazov on Facebook as well as on Twitter (@JamieGlazov) to strengthen his social media strength in the face of the Left\'s vicious war on free speech. Thank you!]',
        'Whom do you consider to be the most corrupt Democrat Politician?'
        ]

    for phrase in phrases_to_delete: 
        text = text.replace(phrase,"")


    #specific junk

    #remove terms and services 
    text = re.sub('Please adhere to our commenting policy to avoid .*? Follow these instructions on registering', '', text, flags=re.DOTALL)

    #remove author text
    text = re.sub('Judi McLeod is an award-winning.*?Rush Limbaugh, Newsmax\.com, Drudge Report, Foxnews\.com\.', '', text, flags=re.DOTALL)

    #remove social media 
    text = re.sub('[0-9]+ SHARES Facebook Twitter', '', text)
    text = re.sub('[0-9][.][0-9]k SHARES Facebook Twitter', '', text)

    #remove poll
    text = re.sub('take our poll.*?Email [*]', '\n', text, flags=re.DOTALL)

    #remove junk validation text 
    text = re.sub('(Phone |Email |Comments |Name )This field is.*?should be left unchanged[.]','', text)
    text = re.sub('Completing this poll grants you.*?Privacy Policy and Terms of Use[.]',"", text, flags=re.DOTALL)

    text = re.sub('Patrick J[.] Buchanan needs no introduction to VDARE[.]COM.*?Patrick J[.] Buchanan is the author of','', text, flags=re.DOTALL)
    text = re.sub('See Peter Brimelow.s review.*?Pat Buchanan.s Nixon Book','', text)



    #youtube
    text = re.sub('Click below to Subscribe to.*?YouTube channel!','', text)
    
    #misc
    
    text = re.sub('This field is for validation purposes and should be left unchanged[.]', '', text, flags=re.DOTALL)
    text = re.sub('take our poll - story continues below', '', text, flags=re.DOTALL)



    #generic
    text = clean_text(text)


    return text 