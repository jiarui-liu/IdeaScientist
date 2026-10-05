"""LaTeX source to plain text.

Strips the markup that would otherwise dominate a paper's ``full_text``:
preamble, figures, tables, and math environments go, prose and section
headings stay.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)




















def _clean_latex(text: str) -> str:
    text = re.sub(r"(?m)^%.*$", "", text)
    text = re.sub(r"(?s)\\begin\{comment\}.*?\\end\{comment\}", "", text)

    text = re.sub(r"\\(?:section|chapter)\{([^}]*)\}", r"\n\n## \1\n", text)
    text = re.sub(r"\\subsection\{([^}]*)\}", r"\n\n### \1\n", text)
    text = re.sub(r"\\subsubsection\{([^}]*)\}", r"\n\n#### \1\n", text)
    text = re.sub(r"\\paragraph\{([^}]*)\}", r"\n**\1** ", text)

    text = re.sub(r"\\(?:textbf|mathbf)\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:textit|emph|mathit)\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:texttt|mathtt)\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:text|mathrm)\{([^}]*)\}", r"\1", text)

    text = re.sub(r"\\(?:cite|citep|citet|citeauthor)\{[^}]*\}", "[citation]", text)
    text = re.sub(r"\\(?:ref|eqref|autoref|Cref|cref)\{[^}]*\}", "[ref]", text)
    text = re.sub(r"\\label\{[^}]*\}", "", text)

    text = re.sub(r"(?s)\\begin\{figure\}.*?\\end\{figure\}", "[Figure]", text)
    text = re.sub(r"(?s)\\begin\{table\}.*?\\end\{table\}", "[Table]", text)

    text = re.sub(r"\\(?:usepackage|documentclass|bibliography|bibliographystyle)"
                  r"(?:\[[^\]]*\])?\{[^}]*\}", "", text)
    text = re.sub(r"\\(?:newcommand|renewcommand|def)(?:\{[^}]*\})+(?:\[[^\]]*\])?(?:\{[^}]*\})*", "", text)
    text = re.sub(r"\\begin\{document\}|\\end\{document\}|\\maketitle", "", text)

    text = re.sub(r"\\item\b", "  - ", text)
    text = re.sub(r"\\begin\{(?:itemize|enumerate|description)\}", "", text)
    text = re.sub(r"\\end\{(?:itemize|enumerate|description)\}", "", text)

    text = re.sub(r"\\begin\{(?:abstract)\}", "\n## Abstract\n", text)
    text = re.sub(r"\\end\{(?:abstract)\}", "\n", text)
    text = re.sub(r"\\begin\{(?:equation|align|gather|multline)\*?\}", "\n[Equation: ", text)
    text = re.sub(r"\\end\{(?:equation|align|gather|multline)\*?\}", " ]\n", text)

    text = re.sub(r"\\[a-zA-Z]+(?:\[[^\]]*\])?\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()




