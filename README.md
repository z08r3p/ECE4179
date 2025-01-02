java c
ECE4179 - Neural Networks and Deep Learning 
Deep   Learning   and   Neural   Networks 
Assignment
Calculation Exercises - General Comments 
For   the   calculation   exercises,   ensure   that   you   document   any   calculations/working-out.   You   can   use   a   word editor to   format your   answers   properly.   You will   receive   a   0   if you   do   not   provide   calculations/working-out.
Calculation Exercise 1: Multilayer Perceptron (MLP) 
1.    Consider   the   MLP   network   in   Figure 1. Table 1 shows      the   value   of   each   of   the   parameters   of   the network.   The   activation   functions   for   all   neurons   are   shown   in   Table 2. The   neurons   in   the   input   layer,   i.e.   neuron   1   and   neuron   2,   do   not   use   any   activation   function   and   simply   pass   in   the   input   to   the network.      All   the   neurons   in   the   hidden   layer   use   a   ReLU   activation   function,    i.e.,   a3   (x)   =   a4   (x)   =   a5   (x) = ReLU(x) = max(0,   x).    The   neuron   in   the   output   layer   uses   a   Sigmoid   activation   function,   i.e.,   a6   (x)   = σ(x) =   1/   (1 + exp(−x)).    For   cross   entropy   loss, ensure you use loge.    Answer   the   following questions.

Figure   1:   An   MLP   with   one   hidden   layer   (Question 1).

Table   1:   Parameter   values   of   the   MLP   (Figure 1).

Table   2:   Activation   functions   of the   MLP   (Figure 1).
1.1. [4   points]    Compute   the   output   of   the   network   for x =   (x1   ,   x2   )T    =   (1,   2)T
1.2. [2   points]    Assume   the   label   of x =   (x1   ,   x2   )T    =   (1,   2)T      is   y   = 0.   If   we   use   the   Binary   Cross   Entropy (BCE)   loss   to   train   our   MLP,   what   will   be   the   value   of the   loss   for   (x,   y)?
1.3. [2   points]    Now   assume   the   label   of x =   (x1   ,   x2   )T    =    (1,   2)T      is   y   =   1.    For   BCE   loss,   what   will   be the   value   of   the   loss   for      (x,   y)?    Do   you   expect   the   loss   to   be   bigger   or   smaller   compared   to   the previous   part?   Why?   Explain   your   answer   and   your   observation.
1.4. [6   points]    Assume   the   learning   rate   of   the   SGD   is   lr   = 0   .1.    For   a   training   sample x =   (x1   ,   x2   )T      =   (1,   2)T      and   y   = 0,   obtain   the   updated   value   of   w3   ,6   .
1.5. [6   points]    Using      the      assumptions   from   the   previous   part      (i.e.,    the      learning   rate   of   the    SGD   is lr   =   0.1,   the   training   sample   is x =   (x1   ,   x2   )T    =    (1,   2)T      and   y   =   0),   obtain   the   updated   value   of w2   ,5   .
Calculation Exercise 2: Activation Function 
2.    [10   points]    Consider   the   following   activation   function:

We stack 1,000 of z, to form.  where ◦ denotes function composition. What would be the response of z1000 to x ∈ R? You need to discuss the behaviour of z1000(x) for x ∈ R. We plot the activation function in Equation (1) along with the 45-degree line in Figure 2 for your convenience.

Figure   2:   Activation   function   for   Question 2.
Coding Exercises - General Comments You   will   code   up   these   exercises   with   the   provided   skeleton   notebooks.   The   results   and   discussions   you   obtain   from   the   notebook   can   stay   within   the   notebook   without   going   into   a   separate   PDF.   Each   task   will   have   its   own   discussion   questions.
Linear and Logistic Regression 
3.    [25   points]    You   are   probably   thinking,   these   models   again!    Okay   you   are   right,   but   there   will   be   addi-   tional   ideas   presented   here.In   the   first   subtask,   you   will   explore   the   effects   of outliers   in   your   dataset   and   how   that   can   affect   your   model   performance.    You   will   apply   a   weighted   linear   regression   model   so   that   the   resulting   model   is   more   robust.In   the   second   subtask,   you   will   be   exploring   ways   to   model   non-linearities   in   the   dataset   with   a   linear   regression   model   by   transforming   your   inputs   to   a   linearised   space   and   performing   classification   in   the transformed   space   with   a   technique   called   decay   learning   rate.
Task 1.1. Weighted linear regression In   this   coding   exercise, we   study   the   iterative   weighted   linear   regression   model.    In   some   problems,   every data   point   might   not   have   the   same   importance.    Having   weights   associated   to   samples   will   provide   us with   a   principal   way   to   model   such   problems.      Consider   a   data   set   in   which   each   data   point   (xi   ,   yi   ),   xi      ∈   Rn   ,   yi      ∈ R   is   associated   with   a   weighting   factor   0   < αi ≤ 1. Define the loss of a linear model with parameters w ∈ Rn as the weighted sum-of-squares error:

It   can   be   shown   that   the   optimal   weights   can   be   obtained   as:

Here,   X   is   a   matrix   of   size   m   ×   n   where   every   row   is   one   input   sample   (i.e.,   row   i   in   X   is   xi   ).    Similarly,   Y   is   an   m   dimensional   vector   storing   yi    and   A   is   a   diagonal   matrix   of size   mm   with   A[i,i] = αi.The   purpose   of this   task   is   to   train   an   iterative   linear   regression   model   that   will   weigh   the   samples   as   the   model   is   being   trained.    You   need   to   develop   the   mechanism   to   weigh   each   sample   as   the   model   is   being   iteratively   trained.   Here,   the   diagonal   matrix   A   is   updated   by   using   the   following   equation:

where   σ   is   a   hyperparemeter   you   will   tune.
You   are   supposed   to   implement   a   weighted   linear   regression   model   to   explain   the   data.    The   data   for   this question has two sets,   a training   set    (X trn   , Y trn(no)isy   )   and   a validation   set    (X val, Y val   ).   You   are supposed to use only the training   set    (X trn   , Y trn(no)isy   ) to   train   your   model.    Note that the   training   set   is   noisy   (i.e., Y trn(no)isy    is   noisy).   You   will   use   the   validation   set   to   evaluate   your   model   and   as   you   might   have   noticed, the   validation   set   is   noise   free.Since   in   this   case,   the   weight   of   each   sample   is   not   provided,   you   need   to   develop   a   mechanism   to   approximate   how   noisy   a   sample   is.    One   way   of   doing   this   is   to   first   fit   a   linear   regression   model   to   your data   and   then   check   and   see   which   samples   your   model   cannot   explain   well   (i.e.,   the   prediction   is   not good).   For   such   samples,   you   need   to   devise   a   way   to   assign   a   reduced   sample   weight.   Once   you   attain   the   weights   according   to   the   error   of   your   predictions,   you   can   fit   a   weighted   linear   regression   model   and repeat   this   process   a   few   times   till   you   obtain   a   model   that   can   explain   the   validation   data   well.
Implement   the   aforementioned   strategy   and   report   your   error   on   the   validation   set.   For   your   convenience,   a   sketch   of a   possible   solution   is   provided   in   Algorithm 1. 
Hints. 
• You   will   craft   new   non-linear   features.
• The   error   of the   validation   set   is

Here,   ˆ(y)i(val)   = wxi(v)al    is   the   prediction   of your   model   for   a   validation   sample   and   m   is   the   number   of
samples   within   your   validation   set.
• You   have   to   tune σ yourself.
• You   have   to   choose   a   stopping   condition.
Algorithm 1: Iterative   Weighted   Linear   Regression
Data: The   training   set X trn ∈ Rm ×n , Y trn(no)isy ∈ Rm
Data: The   validation   set X val      ∈ Rp ×n , Y val      ∈ Rp
Result: The   parameters   of the   optimal   model w*    ∈ Rn
A ←   1;
for iter   in   max-iter do 
w ← Fit a weighted linear regression model using Xtrn,Ynoisytrn , A ; /* Use ?? and note that
A is a diagonal matrix keeping the sample weights ai */
yˆi = w⊤xtrni;
/* Evaluate your model on all training samples */
aˆi = exp
end 
w*    ←   the   model with   the   minimum   validation   error
Task 1.2. Using non-linear features and learning rate decay for better classification You   have   been   given   a   set   of   samples   x   =   (x1   ,   x2   )T      ∈   R2   .    In   this   task,   you   will   apply   the   knowledge you   learned   from   previous   labs   to   train   a   logistic   model   via   Gradient   Descent   and   try   to   improve   the   performance   by   implementing   a   decaying   learning   rate.
Decaying   learning   rate   can   be   simply   implemented   as:lrnew      =   (1 − α) *   lrold   ,                                                                                                          代 写ECE4179 Deep Learning and Neural Networks AssignmentPython
代做程序编程语言                                                                                (6)
where   0   < α < 1 is a hyperparameter. In this task, we use α = 0.1% for the learning rate decay.You   will   follow   the   standard   procedure   of first   loading   in   your   data   and   visualising   it.    From   there,   you   will apply nonlinear transformations to your data via GD. You can then visualise your   decision   boundary   and   write   a   decaying   learning   rate   functino   to   further   improve   your   model.You   can   reuse   functions sigmoid, compute loss and grad and predict from   your   previous   labs   for   this   task.    Plot   the   decision   boundary   for   above   on   test   data.       Compute   the   accuracy   of   the   resulting   model   on   the   test   data   (X test,   y test).
Task 2. Denoising autoencoder (DAE) with Face Data Whenever   you   measure    a   signal,    noise    creeps    in.       Denoising    (aka    noise    reduction)    is    the    process    of   removing   noise   from   a   signal   and   is   a   profound   and   open   engineering   problem.   In   this   exercise,   you   will learn   and   implement   a Denoising AutoEncoder   (DAE)   to   remove   additive   random   noise   from   images.Autoencoders. An   autoencoder   is   a   neural   model   that   is   trained   to   attempt   to   copy   its   input   to   its   output.      Internally,   it   has   a   hidden   layer z that   describes   a   code   used   to   represent   the   input.      To   be   precise,   an   autoencoder   realizes   two   functions/networks,   a   function   fenc    to   transform   the   input x ∈ Rn   to   a   new   representation z ∈   Rm   ,   followed   by   a   function   fdec   ,   which   converts   the   new   representation z back   into   the   original   representation.    We   call   these   two   functions   the   encoder   and   the   decoder      (see Figure 3 for   an   illustration).You may ask yourself if an autoencoder succeeds in   simply   learning   to   set   fdec   (fenc   (x))   = xeverywhere,   then   what    is   it   useful    for?       The      short      answer      to      that      is,    the      new      representation z learned      by      the autoencoder   captures   useful   information   about   the   structure   of   the   data   in   a   compact   form   that   is   friendly to ML algorithms.    But that is not all.   In deep   learning,   we   can   borrow   the   idea   of autoencoding   and   design   many   useful   solutions,   denoising   being   one.DAE. In   a   DAE,   instead   of   showing x to   the   model,   we   show   a   noisy   input   as   x(˜) = x +   ϵ   .    The   noise   ϵ   is task   specific.   If   you are   working   with   MRI   images,   this   noise   should   model   the   noise   of   an   MRI   machine.   If you   are   working   on   speech   signals,   the   noise   could   be   the   babble   noise   recorded   in   a   cafe.    What   you will   ask   your   DAE   to   do   is   now   to   generate   clean   samples,   i.e.   x(ˆ) = x (see   Figure 3 for   an   illustration).So   in   essence,   knowing   about   the   problem   and   associated   noise   will   enable   you   to   simulate   it.   This   will give   you   a   very   task-specific   solution,   which   is   in   many   cases   desirable.
Figure   3: Top. An   autoencoder   is   comprised   of   two   subnetworks,   an   encoder   and   a   decoder.    In   its   vanilla form, the goal is to   extract   useful   structure   and   information   about   the   input. Bottom. In   a DAE, you   feed   the   network   with   a   noisy   input   and   train   the   model   to   produce   clean   outputs.
4.    [25   points]    In   this   task,   you   will   implement   a   DAE   to   reduce   the   amount   of   noise   within   the   dataset.   Table 3 shows   the   hyperparameters   and   network   architecture   that   you   will   be   using.
Table   3:   Hyperparameters   for   the   DAE   model.   Note   that   some   of   the   hyperparameters   have   been   replaced with   ”   ???   ”You   will   need   to   Compute   Peak   Signal-to-Noise   Ratio   (PSNR) for   test   dataset.   The   PSNR   isa   commonly used   metric   to   measure   the   quality   of areconstructed   or   denoised   signal   or   image.   The   PSNR   is   calculated as   the   ratio   of   the   peak   signal   power   to   the   noise   power, typically   measured   in   decibels   (dB).   PSNR   can be   defined   as

Where:
• PSNR   is   the   Peak   Signal-to-Noise   Ratio   in   dB.
• MAX   is   the   maximum   possible   pixel   value   (e.g.,   255   for   an   8-bit   image).
•    MSE   is   the   Mean   Squared   Error   between   the   original   and   the   reconstructed   (or   denoised)   image.   It’s   calculated   as   the   average   of squared   pixel-wise   differences   between   the   two   images.
You   will   be   asked   to   show   the   PSNR   values   for:
1.    Between   the   noisy   image   and   original   image
2.    Between   the   reconstructed   image   and   original   image
We   provide   you   a   handy   class   torchmetrics.image.PeakSignalNoiseRatio   for   PSNR   calculation.Note:   To   create   the   noisy   data, you   will   inject   random   noise   into   the   data   samples   during   the   training step within   your   network.      This   is   shown   in   the   skeleton   code.      You   will   also   need   to   reshape   the   output   predictions   to   visualize   the   output   image.
Task 3. Speech Command Recognition 
5.    [20   points]    In   this   final   task   of   your   assignment,   you   will   be   loading   a   speech   recognition   model   from the   following   options:
• Word2Vec2
• Whisper
• HuBERT
And   using   it   to   extract   feature   representations   from   audio   files.    The   goal   is   to   train   a   model   that   can   recognize   ten   different   voice   commands,   including   ”Yes”,   ”No”,   ”Down”,   and   more.
To   accomplish   this   task,   you   will   follow   these   steps:
• Dataset: You   will   work   with   the   Speech   Commands   dataset,   which   consists   of   audio   recordings   of various   command   words.    The   dataset   is   divided   into   three   folders:   Train,   Validation,   and   Test.   The   Train   folder   contains   audio   files   for   training   the   models,   the      Validation      folder   contains   audio files   for   hyperparameter   tuning,   and   the   Test   folder   contains   audio   files   for   final   predictions.
• Preprocessing: You   will   preprocess   the   audio   data   by   extracting   meaningful   features   using   your   model   of choice.   You   can   choose   a   pre-trained   model   that   has   been   trained   on   labeled   speech   data   and   can   effectively   capture   spoken   language   patterns   and   features   effectively.    The   pre-processing   model   normally   comes   with   the   pre-trained   model   that   we   have   listed   above.
• Model Training: You   will   train   a   simple   MLP    (Multi-Layer   Perceptron)   model   to   predict   the   labels   of the   voice   commands.   The   MLP   model   will   take   the   extracted   features   as   input   and   learn   to   classify   the   voice   commands   into   ten   different   classes.
• Evaluation: You   will   evaluate   the   performance   of your   trained   model   using   accuracy   as   the   eval-   uation   metric.   Accuracy   measures   the   percentage   of correctly   predicted   labels.
• Testing: Finally,   you   will   make   predictions   on   the   test   dataset   using   your   trained   model   and   submit   your   results   to   Kaggle   (instruction   is   provided   in   the   notebook).
• Marking: 
– 10/20   points   are   allocated   to   the   notebook   quality.
– 10/20 points   are   allocated   to   the   Kaggle   Competition   results.   You   need   to   achieve   at   least   85%   accuracy   on   the   test   set.    The   competition   marks   will   be   linearly   awarded   from   0   to   5   points   according   to   accuracy   from   85%   to   highest   accuracy   in   this   competition.
Kaggle    Competition: The      link   to   the      competition      is   provided      here: https://www.kaggle.com/ competitions/ecse-4179-5179-6179-2024-s-2-assignment-1 
• Your   student   email   has   been   registered   for   the   competition.
•   You   should   create   aKaggle   account with your   student email   address   and then   attempt   the   compe-   tition.
• You   have   access   to   the   dataset   from Data tab   in   the   Kaggle   page.By   completing   this   task,   you   will   gain   hands-on   experience   in   speech   recognition   using   the   pre-trained   speect   recognition   models   and   learn   how   to   train   a   model   to   recognize   voice   commands.       This      skill has   wide-ranging   applications,   including   voice-controlled   systems   for   smart   home   devices,   navigation systems,   and   automotive   applications.




         
加QQ：99515681  WX：codinghelp  Email: 99515681@qq.com
