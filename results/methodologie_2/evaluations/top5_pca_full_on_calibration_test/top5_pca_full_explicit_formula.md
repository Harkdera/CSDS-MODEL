# Formule explicite du modèle final PCA 5 composantes

Le modèle final est ajusté sur la cible :

\[
\hat z = \widehat{\log(e-c)}
\]

avec :

\[
\hat e = c + \exp(\hat z)
\]

et, ensuite,

\[
\hat d = \text{reconstruit à partir de l'équation du pic CSDS}
\qquad\text{et}\qquad
\hat b = \hat d - a.
\]

## Variables d'entrée

\[
\begin{aligned}
x_1 &= \tau_r / \tau_p \\
x_2 &= u_p / u_r \\
x_3 &= u_r \\
x_4 &= u_r u_p \\
x_5 &= \tau_p / u_p \\
x_6 &= \tau_r \\
x_7 &= \tau_r / u_p \\
x_8 &= \tau_p / u_r \\
x_9 &= \tau_p \tau_r \\
x_{10} &= \tau_r / u_r
\end{aligned}
\]

## Standardisation

\[
s_i = \frac{x_i-\mu_i}{\sigma_i}
\]

avec :

| Variable | \(\mu_i\) | \(\sigma_i\) |
|---|---:|---:|
| \(x_1\) | 0.7557561152 | 0.1965951865 |
| \(x_2\) | 0.2764020875 | 0.2078702697 |
| \(x_3\) | 5.2151718582 | 4.7388199848 |
| \(x_4\) | 10.6148636537 | 24.0821983183 |
| \(x_5\) | 6.4340743926 | 7.8183063472 |
| \(x_6\) | 4.1514763376 | 5.3795026659 |
| \(x_7\) | 4.9939410410 | 6.4791531278 |
| \(x_8\) | 1.3496149780 | 1.4436255120 |
| \(x_9\) | 51.9466935663 | 90.1535920914 |
| \(x_{10}\) | 1.1245439948 | 1.3010084027 |

## Composantes principales

\[
\begin{aligned}
\mathrm{PC}_1 &= 0.2177072822\,s_1 + 0.1267627406\,s_2 - 0.1341183521\,s_3 - 0.0770200260\,s_4 \\
&\quad + 0.3275696140\,s_5 + 0.4051146748\,s_6 + 0.3574044208\,s_7 + 0.4260795048\,s_8 \\
&\quad + 0.3840688093\,s_9 + 0.4294236413\,s_{10}
\end{aligned}
\]

\[
\begin{aligned}
\mathrm{PC}_2 &= 0.2897159103\,s_1 + 0.4924304460\,s_2 + 0.3741257161\,s_3 + 0.4992636280\,s_4 \\
&\quad - 0.3446549802\,s_5 + 0.2040642274\,s_6 - 0.2830695495\,s_7 - 0.0114912714\,s_8 \\
&\quad + 0.2028276633\,s_9 + 0.0501396149\,s_{10}
\end{aligned}
\]

\[
\begin{aligned}
\mathrm{PC}_3 &= -0.3364809407\,s_1 - 0.4380940758\,s_2 + 0.5918599917\,s_3 + 0.4437020820\,s_4 \\
&\quad + 0.2803608351\,s_5 + 0.0717791569\,s_6 + 0.2313912300\,s_7 + 0.0289322701\,s_8 \\
&\quad + 0.0948876403\,s_9 - 0.0233940772\,s_{10}
\end{aligned}
\]

\[
\begin{aligned}
\mathrm{PC}_4 &= 0.7557768607\,s_1 - 0.1345478635\,s_2 - 0.0197907800\,s_3 + 0.2671149502\,s_4 \\
&\quad + 0.2177280797\,s_5 - 0.1983866806\,s_6 + 0.3267750698\,s_7 - 0.1578790775\,s_8 \\
&\quad - 0.3329302136\,s_9 - 0.0981991321\,s_{10}
\end{aligned}
\]

\[
\begin{aligned}
\mathrm{PC}_5 &= -0.3855284427\,s_1 + 0.4820616655\,s_2 - 0.3522145507\,s_3 + 0.4920469155\,s_4 \\
&\quad + 0.2895733499\,s_5 - 0.2227718500\,s_6 + 0.1274961481\,s_7 + 0.1489197031\,s_8 \\
&\quad - 0.2836990477\,s_9 + 0.0205328075\,s_{10}
\end{aligned}
\]

Les cinq composantes expliquent ensemble environ 97.31 % de la variance des dix variables d'entrée.

## Régression polynomiale finale sur les 5 composantes

\[
\begin{aligned}
\hat z =\;& -0.0063660393
- 0.0622452474\,\mathrm{PC}_1
- 1.5490314429\,\mathrm{PC}_2
+ 0.0369348318\,\mathrm{PC}_3 \\
&+ 1.0102847327\,\mathrm{PC}_4
- 1.3462814741\,\mathrm{PC}_5
+ 0.0396064744\,\mathrm{PC}_1^2 \\
&+ 0.1489318612\,\mathrm{PC}_1\mathrm{PC}_2
- 0.1396323894\,\mathrm{PC}_1\mathrm{PC}_3
- 0.0625403808\,\mathrm{PC}_1\mathrm{PC}_4 \\
&- 0.1747226566\,\mathrm{PC}_1\mathrm{PC}_5
+ 0.2472911198\,\mathrm{PC}_2^2
- 0.1420365129\,\mathrm{PC}_2\mathrm{PC}_3 \\
&+ 0.2030251616\,\mathrm{PC}_2\mathrm{PC}_4
+ 0.1436469275\,\mathrm{PC}_2\mathrm{PC}_5
+ 0.1021818701\,\mathrm{PC}_3^2 \\
&- 0.1867255981\,\mathrm{PC}_3\mathrm{PC}_4
- 0.0846506248\,\mathrm{PC}_3\mathrm{PC}_5
- 0.1408774406\,\mathrm{PC}_4^2 \\
&- 0.0169098434\,\mathrm{PC}_4\mathrm{PC}_5
- 0.2535996789\,\mathrm{PC}_5^2
\end{aligned}
\]

## Reconstruction finale

\[
\hat e = c + \exp(\hat z)
\]

où \(c\) est obtenu à partir du modèle CSDS, puis :

\[
\hat d = \text{équation du pic CSDS}(\hat e),
\qquad
\hat b = \hat d - a.
\]
