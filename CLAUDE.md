# Claude Code Instructions — Simple Trader R1

## Session Start

At the beginning of every conversation, before doing anything else:
1. Read `README.md` to understand the current state of the project.
2. Think Before Coding. Don't assume. Don't hide confusion. Surface tradeoffs.
3. Before you write the code, plan it out. Explain each step. If something is unclear, ask." This one line doubles the quality of the output. Claude guesses less when he thinks.
4. Write the tests first. Then write code that passes the tests. Then refactor."

5. Touch only what you must. Clean up only your own mess.

    When editing existing code:

        Don't "improve" adjacent code, comments, or formatting
        Don't refactor things that aren't broken
        Match existing style, even if you'd do it differently
        If you notice unrelated dead code, mention it — don't delete it


6. Just the code I asked for. No further explanation, No introduction, or explanation  unless otherwised asked by user.

7. when agents saves models and /or jounalls gets saved the filing and directory system must be clean tidy and intitive - no messy files and folders

## ⚠️ IMPORTANT — For AI Assistants / Low-Context Sessions
> always start any task by refering or carefully reading readme file.

> **Do NOT remove, rename, or stub out any existing function, class, method, callback,
> or file unless the user explicitly asks you to delete it.**

>work through this systematically, starting with a file-by-file plan. Set up the todo list and start working systematically.

> This codebase is large. If you are running low on context or tokens:
> - Edit only the specific lines/functions you were asked to change.
> - Leave all surrounding code untouched.
> - Do not "clean up" imports, remove unused variables, or refactor adjacent code.
> - Do not replace working implementations with `pass` or `raise NotImplementedError`.
> - If unsure whether something is used, assume it IS used and leave it alone.
> - before you change the code read Readme.md and if you change code update the READme.md for the change that made

 > -  work through this systematically, starting with a file-by-file plan. Set up the todo list and start working systematically.  Tidy up the codes without losing functionalities or breaking the codes, make sure the speed and accuracy of the code is achieved - double checks before you go to next part of the code. everything should be clean code, fast  and performance to  highest quality

>- make sure you remove the redudant codes, variables and tidy up the code without losing functionalities or breaking the codes.  make sure codes are clean, readable, professional and comply with highest coding standards.
> - once you review the code, make a note of all the bottleknecks in the codes and let user know the effect of improving on the speed , perforamce and stabilty or anyother metric in the code execution.
>- update the Readme file once finish 
---



## Project Overview

RL trading agent (RecurrentPPO / LSTM) trained on ES/NQ futures using Stable-Baselines3.
See README.md for full architecture, file layout, and training pipeline details.
""  https://raw.githubusercontent.com/forrestchang/andrej-karpathy-skills/main/CLAUDE.md 
