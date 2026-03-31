# PyBGE

This is a small Python package designed to download then tabulate your hourly energy usage data from the BGE (Baltimore Gas and Electric) website, and then do a little analysis with it to understand and predict household energy usage. 

I made this code as a little personal project, because we had done some energy efficiency upgrades to our house, and I wanted to understand how much of an impact these were having on our bills. The BGE  account website provides very useful hour-by-hour usage plots for electricity and gas. Cool! However, they don't provide a way to download that usage data in bulk.

`PyBGE` gets around this problem by logging into the BGE website on your behalf, navigating to the hourly usage plot for a given day, *screenshotting* it that plot, parsing the screenshot to get your hour-by-hour energy usage, and then repeating that process for every day within a range your define.`PyBGE` then tabulates all of this data, and cross-references it with publicly available weather data.

`PyBGE` also provides functionality to let you use this data for some analysis!

Firstly, `PyBGE` can apply a range of (fairly basic) machine learning models to the usage and weather data, to train a model to predict your usage based on conditions. You can define training versus comparison date ranges, to see if your more recent usage is different from what the model would predict based on your past usage patterns. This allowed me to figure out how much my energy improvements were saving from my bills!

Secondly, `PyBGE` can combine weather forecasts and historical weather data, to use its model of your usage to predict how much energy you are likely to use in the near future (I personally find this more accurate than BGE's own estimate of what my usage will be over a given billing period).

## Big Important Warnings

I made this code for personal use. Please do not use it to DoS the BGE website, or do anything else silly/nefarious. I appreciate BGE providing the hourly usage data they do, even though I *really* wish they would allow us to download it in bulk.

`PyBGE` needs you to provide, in plaintext, the email and password of your BGE account, so that it can log in to it and do its job. It is a **bad idea** to put your email and password into some random Python code you got off GitHub.

If you have MFA enabled on your BGE account (*and you should)*, then `PyBGE` will also ask you to input the MFA code that BGE sends you when it tries to log in to your account. **This is a TERRIBLE idea**. You absolutely *should not* type your MFA code into some random Python code you got off GitHub! Why would you even do that? I could be a terrible person with nefarious plans! For heaven's sake, I chose to spend part of my limited time on this Earth writing code to scrape energy usage data off the website of a utility company - I clearly don't have good judgement! You absolutely should not trust my software with your MFA code.

I wrote `pyBGE` myself; however it was very much tailored to my own setup. I therefore used an LLM to refactor it into the more generic package structure you see here. I have a gaming PC I use regularly, so I judged that my net "compute and energy usage for silly purposes" budget was not meaningfully impacted by this use of an LLM.

## Installation & Requirements

## Quick Guide to Using PyBGE
