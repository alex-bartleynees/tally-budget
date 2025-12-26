# Tally

**Let AI classify your transactions.**

## The Problem

Bank transactions are a mess:
```
WHOLEFDS MKT 10847 SEATTLE WA
AMZN MKTP US*2K7X9
SQ *JOES COFFEE SEATTLE
CHECKCARD 0415 COSTCO WHSE #1234
```

And even when your bank *does* categorize them, their categories aren't *your* categories. They say "Shopping" but you want to know it's "Kids > Clothing" vs "Home > Furniture". They lump all restaurants together but you want to separate "Coffee" from "Fast Food" from "Fine Dining".

Building a custom categorization system traditionally required:
- Massive merchant databases (expensive, still incomplete)
- Complex rules engines or a CMS (over-engineered)
- Manual classification of thousands of transactions (tedious)
- Accepting whatever categories your bank decided on (useless)

## The Solution

LLMs have read the internet. They *know* what "WHOLEFDS MKT" is. They understand that "SQ *" means Square payment at a small business. They can tell "COSTCO WHSE" is groceries but "COSTCO GAS" is fuel.

Better yet, you can use natural language to tell the AI what rules apply to *your* specific situation:

```
"ZELLE payments to Sarah are for babysitting - categorize as Childcare"
"Anything at COSTCO that includes GAS is fuel, otherwise it's groceries"
"I want to track coffee shops separately from other restaurants"
```

The AI understands your intent, writes the pattern-matching rules, and saves them to a simple CSV file. No complex CMS. No rules engine. Just a text file you can version control, edit by hand, or let the AI maintain.

**Tally** is designed to work *with* an AI assistant. You describe what you want, the AI figures out the patterns, and Tally executes the categorization at scale.

## How It Works

```
You: "Help me categorize my spending"

AI: *runs tally discover --format json*
AI: "I found 47 unknown merchants. Let me classify them..."
AI: *identifies each merchant, adds rules*
AI: *runs tally run --summary*
AI: "Done. Only 3% unknown now. Here's your spending breakdown..."
```

The AI handles the *understanding*. Tally handles the *execution*.

## Quick Start

```bash
# Install
uv tool install git+https://github.com/davidfowl/tally

# Initialize a budget directory
tally init ./my-budget

# Add your bank exports to my-budget/data/
# Configure my-budget/config/settings.yaml

# Let an AI assistant classify your transactions
# Or run directly:
tally run
```

## Commands

```bash
tally init [dir]           # Create a new budget directory
tally run                   # Analyze and generate report
tally run --summary         # Quick summary, no HTML
tally discover              # Find unknown merchants
tally discover --format json # JSON output for AI agents
tally inspect <file.csv>    # Examine CSV structure
tally diag                  # Debug configuration issues
```

## Why This Approach?

**Before LLMs:** You needed a commercial service with merchant databases, or spent hours manually categorizing, or accepted bad data.

**With LLMs:** Point an AI at your transactions. It reads "AMZN MKTP US*2K7X9" and knows that's Amazon Marketplace. It sees "SQ *JOES COFFEE" and knows that's a coffee shop using Square. It understands context humans built those merchant databases with—but now it's instant and free.

**Tally's role:** Once the AI identifies merchants, Tally executes the categorization at scale. Regex patterns run in milliseconds. The 720 built-in rules handle common merchants automatically. Your custom rules handle the rest.

## The Workflow

1. **Export** your bank/credit card statements
2. **Run** `tally discover --format json` to find unknowns
3. **Ask** your AI assistant to classify them
4. **Review** the rules it adds to `merchant_categories.csv`
5. **Run** `tally run` to generate your spending report
6. **Iterate** until unknowns are minimal

Most users get to 95%+ categorization in one session with AI assistance.

## Configuration

```yaml
# config/settings.yaml
year: 2025
data_sources:
  - name: AMEX
    file: data/amex.csv
    type: amex
  - name: Chase
    file: data/chase.csv
    format: "{date:%m/%d/%Y},{description},{amount}"
```

```csv
# config/merchant_categories.csv
Pattern,Merchant,Category,Subcategory
WHOLEFDS,Whole Foods,Food,Grocery
UBER\s(?!EATS),Uber,Transport,Rideshare
UBER\s*EATS,Uber Eats,Food,Delivery
```

## For AI Agents

When you run `tally init`, it creates `AGENTS.md` with detailed instructions for AI assistants. The key commands:

- `tally discover --format json` - Get unknown merchants as structured data
- `tally diag` - Debug why rules might not be loading
- `tally inspect <file>` - Understand CSV format before configuring

The AI reads the merchant descriptions, understands what they are, and writes the regex patterns. Tally does the rest.

## License

MIT
