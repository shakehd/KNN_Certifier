% --------------------------------------------------------------------
%  @LaTeX-class-file{
%     filename        = "Dissertate.cls",
%     codetable       = "ISO/ASCII",
%     keywords        = "LaTeX, Dissertate",
%     docstring       = "Class for a dissertation."
% --------------------------------------------------------------------

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{Dissertate}[Dissertate Class]
\LoadClass[12pt,twoside,openright,a4paper]{book}

% Options
\RequirePackage{etoolbox}
\RequirePackage[a-2b,mathxmp]{pdfx}[\today]

% Line spacing: dsingle/ddouble
%   Whether to use single- or doublespacing
\newtoggle{DissertateSingleSpace}
\toggletrue{DissertateSingleSpace}
\DeclareOption{dsingle}{
    \toggletrue{DissertateSingleSpace}
    \ClassWarning{Dissertate}{Single-spaced mode on.}
}
\DeclareOption{ddouble}{\togglefalse{DissertateSingleSpace}}

\ProcessOptions\relax

% Line Spacing
%   Define two line spacings: one for the body, and one that is more compressed
\iftoggle{DissertateSingleSpace}{
	\newcommand{\dnormalspacing}{1.2}
	\newcommand{\dcompressedspacing}{1.0}
}{
	\newcommand{\dnormalspacing}{2.0}
	\newcommand{\dcompressedspacing}{1.2}
}

% Block quote with compressed spacing
\let\oldquote\quote
\let\endoldquote\endquote
\renewenvironment{quote}
	{\begin{spacing}{\dcompressedspacing}\oldquote}
	{\endoldquote\end{spacing}}

% Itemize with compressed spacing
\usepackage{enumitem}
\let\olditemize\itemize
\let\endolditemize\enditemize
\renewenvironment{itemize}
	{\begin{spacing}{\dcompressedspacing}\olditemize}
	{\endolditemize\end{spacing}}

% Enumerate with compressed spacing
\let\oldenumerate\enumerate
\let\endoldenumerate\endenumerate
\renewenvironment{enumerate}
	{\begin{spacing}{\dcompressedspacing}\oldenumerate}
	{\endoldenumerate\end{spacing}}

% Text layout
\RequirePackage[outer=1.2in, inner=1.2in, twoside, a4paper]{geometry}
\usepackage{ragged2e}
\usepackage{graphicx}
\usepackage{tabularx}
\parindent 12pt
\RequirePackage{lettrine}
\RequirePackage{setspace}
\RequirePackage{verbatim}
\RequirePackage{amsmath, bm}
\RequirePackage{amsthm}
\RequirePackage{galois}
\RequirePackage{mathtools}
\DeclarePairedDelimiter{\abs}{\lvert}{\rvert}
\RequirePackage{siunitx}

% Fonts
\RequirePackage{color}
\RequirePackage{colortbl}
\RequirePackage{hyperref}
\RequirePackage{systeme}
\renewcommand{\chapterautorefname}{Chapter}
\renewcommand{\sectionautorefname}{Section}
\renewcommand{\subsectionautorefname}{Subsection}
\RequirePackage{url}
\Urlmuskip = 0mu plus 1mu
\RequirePackage[T1]{fontenc}
\RequirePackage[libertine]{newtxmath}
\RequirePackage{mathspec}
\AtBeginEnvironment{tabular}{\addfontfeature{RawFeature=+tnum}}
\widowpenalty=300
\clubpenalty=300
\setromanfont[Ligatures={Common,TeX}, Scale=1.0]{Linux Libertine O}
\newfontfamily{\smallcaps}[RawFeature={+c2sc,+scmp}]{Linux Libertine O}
\setsansfont[Scale=MatchLowercase, BoldFont={Lato Bold}]{Lato Regular}
\setmonofont[Scale=MatchLowercase]{Source Code Pro}
\RequirePackage[labelfont={bf,sf,normalsize,singlespacing},
                textfont={sf,normalsize,singlespacing},
                justification={justified},
                singlelinecheck=false,
                margin=0pt,
                figurewithin=chapter,
                tablewithin=chapter]{caption}
\renewcommand{\thefootnote}{\arabic{footnote}}
\RequirePackage{microtype}

% Theorems
\newtheoremstyle{normalstyle}%                    % Name
    {}%                                           % Space above
    {}%                                           % Space below
    {}%                                           % Body font
    {}%                                           % Indent amount
    {\bfseries}%                                  % Theorem head font
    {.}%                                          % Punctuation after theorem head
    { }%                                          % Space after theorem head, ' ', or \newline
    {\thmname{#1}\thmnumber{ #2}\thmnote{ (#3)}}% % Theorem head spec (can be left empty, meaning `normal')
\newtheoremstyle{italicstyle}%                    % Name
    {}%                                           % Space above
    {}%                                           % Space below
    {\itshape}%                                   % Body font
    {}%                                           % Indent amount
    {\bfseries}%                                  % Theorem head font
    {.}%                                          % Punctuation after theorem head
    { }%                                          % Space after theorem head, ' ', or \newline
    {\thmname{#1}\thmnumber{ #2}\thmnote{ (#3)}}% % Theorem head spec (can be left empty, meaning `normal')

% Tikz
\usepackage{tikz}
\usetikzlibrary{shadows}
\usetikzlibrary{patterns,intersections}
\usepackage{pgfplots}
\usepgfplotslibrary{fillbetween}

\pgfplotsset{compat=newest}
\pgfplotsset{plot coordinates/math parser=false}
\newlength\fheight
\newlength\fwidth
\usepackage{booktabs}
\usepackage{multirow}
%\usepackage{multicolumn}
\usepackage{subcaption}
\usetikzlibrary{patterns,decorations.pathreplacing,backgrounds,calc}
\usepackage{afterpage}
\usepackage{tabulary}
\newcommand{\ra}[1]{\renewcommand{\arraystretch}{#1}}

% Pseudocode
\usepackage{algorithm}
\usepackage{algpseudocodex}
\renewcommand\algorithmicthen{}
\@addtoreset{algorithm}{chapter}% algorithm counter resets every chapter
\usepackage{lscape}
\renewcommand{\thealgorithm}{\thechapter.\arabic{algorithm}}% Algorithm # is <chapter>.<algorithm>
\newcommand{\algorithmautorefname}{Algorithm}
\algrenewcommand\algorithmicrequire{\textbf{Input:}}
\algrenewcommand\algorithmicensure{\textbf{Output:}}

% Comments and notes
\usepackage[colorinlistoftodos]{todonotes}

% Headings and headers
\RequirePackage{fancyhdr}
\RequirePackage[medium,bf]{titlesec}% format for the title of the sections
\setlength{\headheight}{15pt}
\pagestyle{plain}
\RequirePackage{titling}

% Front matter
\setcounter{tocdepth}{2}
\usepackage[titles]{tocloft}
\usepackage[titletoc]{appendix}
\usepackage{acronym}
\renewcommand{\cftchapfont}{\bfseries}
\renewcommand{\cftsecleader}{\cftdotfill{\cftdotsep}}

% Endmatter
\renewcommand{\setthesection}{\arabic{chapter}.A\arabic{section}}

% References
\renewcommand\bibname{References}
\RequirePackage[comma,numbers]{natbib}
\renewcommand{\bibnumfmt}[1]{[#1]}
\RequirePackage[palatino]{quotchap}
\renewcommand*{\chapterheadstartvskip}{\vspace*{-0.5\baselineskip}}
\renewcommand*{\chapterheadendvskip}{\vspace{1.3\baselineskip}}
\bibliographystyle{IEEEtranS}

% An environment for paragraph-style section
\providecommand\newthought[1]{%
	\addvspace{1.0\baselineskip plus 0.5ex minus 0.2ex}%
	\noindent\textsc{#1}%
}

% Align reference numbers so that they do not cause an indent
\newlength\mybibindent
\setlength\mybibindent{0pt}
\renewenvironment{thebibliography}[1]
	{\chapter*{\bibname}%
		\@mkboth{\MakeUppercase\bibname}{\MakeUppercase\bibname}%
		\list{\@biblabel{\@arabic\c@enumiv}}
			{\settowidth\labelwidth{\@biblabel{999}}
				\leftmargin\labelwidth
				\advance\leftmargin\dimexpr\labelsep+\mybibindent\relax\itemindent-\mybibindent
				\@openbib@code
				\usecounter{enumiv}
				\let\p@enumiv\@empty
				\renewcommand\theenumiv{\@arabic\c@enumiv}}
		\sloppy
		\clubpenalty4000
		\@clubpenalty \clubpenalty
		\widowpenalty4000%
		\sfcode`\.\@m}
	{\def\@noitemerr
		{\@latex@warning{Empty `thebibliography' environment}}
			\endlist}

% Some definitions
\def\university#1{\gdef\@university{#1}}
\def\department#1{\gdef\@department{#1}}
\def\advisor#1{\gdef\@advisor{#1}}
\def\coadvisorOne#1{\gdef\@coadvisorOne{#1}}
%\def\coadvisorTwo#1{\gdef\@coadvisorTwo{#1}}
\def\coadvisorsUniversity#1{\gdef\@coadvisorsUniversity{#1}}
\def\mastername#1{\gdef\@mastername{#1}}
\def\studentId#1{\gdef\@studentId{#1}}
\def\academicYear#1{\gdef\@academicYear{#1}}
\def\place#1{\gdef\@place{#1}}
\def\degreeyear#1{\gdef\@degreeyear{#1}}

% School color
\definecolor{SchoolColor}{rgb}{0.71, 0, 0.106}% UNIPD red
\definecolor{chaptergrey}{rgb}{0.61, 0, 0.09}% dialed back a little
\definecolor{midgrey}{rgb}{0.4, 0.4, 0.4}

\hypersetup{
	colorlinks,
	citecolor=SchoolColor,
	filecolor=black,
	linkcolor=black,
	urlcolor=SchoolColor,
}

\renewcommand{\frontmatter}{
	\pagenumbering{roman}
	\input{frontmatter/personalize}
	\maketitle
	\copyrightpage
	\dedicationpage
	\acknowledgments
	\abstractpage
    \tableofcontents
    \clearpage

	\begingroup
        \let\clearpage\relax
        \let\cleardoublepage\relax

		% figure listing - required if you have any figures
		\phantomsection
		\addcontentsline{toc}{chapter}{List of figures}
		\listoffigures
        \vspace{3.1cm}

		% table listing - required if you have any tables
        \phantomsection
		\addcontentsline{toc}{chapter}{List of tables}
		\listoftables
        \vspace{3.1cm}

        % algorithm listing - required if you have any algorithms
        \phantomsection
        \addcontentsline{toc}{chapter}{List of algorithms}
        \listofalgorithms
        \vspace{3.1cm}

		% acronym listing - required if you have any acronyms
        \phantomsection
		\addcontentsline{toc}{chapter}{List of acronyms}
		\chapter*{List of Acronyms}
		\input{frontmatter/acronyms}
	\endgroup

    \cleardoublepage
	\pagenumbering{arabic}
}

\renewcommand{\maketitle}{
	\thispagestyle{empty}
	\begin{center}
		\vbox to0pt{\vbox to\textheight{\vfill \includegraphics[width=11.5cm]{resources/unipd-light} \vfill} \vss}
		\begin{figure}
			\centering
			\includegraphics[height=2.5cm]{resources/unipd-bn}
		\end{figure}

		\setstretch{1.5}

		\normalfont{\huge{\bfseries{\@university}}} \\
		\line(1, 0){400} \\
		\scshape{\large{Department of \@department}} \\
		\vspace{3pt}
		\scshape{\large{Master Thesis in \@mastername}} \\

		\vspace{2pt}
		\setstretch{2.5}

		\vspace{10pt}
		\normalfont{\LARGE{\bfseries{\textcolor{SchoolColor}{\@title}}}} \normalsize \\
		\vspace{15pt}

		\setstretch{1.2}
		\vfill

		\begin{normalsize}
			\begin{flushleft}
				\textit{Supervisor} \hfill \textit{Master Candidate} \\
				\vspace{1pt}
				\@advisor \hfill \@author \\
				\@university \\
				\vspace{6pt}
				\textit{Co-supervisor} \hfill \textit{Student ID} \\
				\@coadvisorOne \hfill \@studentId \\
				%\@coadvisorTwo \\
				\@coadvisorsUniversity \\
				\vspace{30pt}
				\centering
				\large{\scshape{Academic Year} \\ \@academicYear}
			\end{flushleft}
		\end{normalsize}
	\end{center}
	\vspace*{\fill}
	\singlespacing
	\newpage
}

\newcommand{\copyrightpage}{
	\phantomsection
	\thispagestyle{empty}
	\vspace*{\fill}
	\noindent \@author : \textit{\@title}, Master Thesis, \copyright \@degreeyear.
	\newpage
}

\newcommand{\dedicationpage}{
	\phantomsection
	\thispagestyle{empty}
	\vspace*{\fill}
	\scshape \noindent \input{frontmatter/dedication}
	\vspace*{\fill}
	\newpage
	\thispagestyle{empty}
	~\newpage
	\rm
}

\newcommand{\acknowledgments}{
	\phantomsection
	\addcontentsline{toc}{chapter}{Acknowledgments}
	\chapter*{Acknowledgments}
	\noindent
	\input{frontmatter/thanks} \\~\\~\\ \noindent \@place, \@degreeyear \hfill \@author
	\vspace*{\fill}
 	\cleardoublepage
}

\newcommand{\abstractpage}{
	\phantomsection
	\addcontentsline{toc}{chapter}{Abstract}
	\chapter*{Abstract}
	\input{frontmatter/abstract}
	\cleardoublepage
}

% Listing
\usepackage{listings}
\lstdefinelanguage{json}{
	string=[s]{"}{"},
	stringstyle=\color{SchoolColor},
	comment=[l]{:},
	commentstyle=\color{black},
	basicstyle=\normalfont\ttfamily,
	numbers=left,
	numberstyle=\scriptsize,
	stepnumber=1,
	numbersep=8pt,
	showstringspaces=false,
	breaklines=true,
	frame=lines
}
