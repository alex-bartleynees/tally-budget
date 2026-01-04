#!/usr/bin/env python3
"""Performance test for rule matching."""

import time
import random
import string
from datetime import date
from tally.merchant_engine import MerchantEngine, MerchantRule


def generate_random_description(length=30):
    """Generate a random transaction description."""
    words = ['AMAZON', 'UBER', 'NETFLIX', 'COSTCO', 'STARBUCKS', 'TARGET', 'WALMART',
             'WHOLE', 'FOODS', 'TRADER', 'JOES', 'SAFEWAY', 'SHELL', 'CHEVRON',
             'DOORDASH', 'GRUBHUB', 'INSTACART', 'APPLE', 'GOOGLE', 'MICROSOFT']

    # Pick 1-3 words and add random suffix
    selected = random.sample(words, random.randint(1, 3))
    suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    return ' '.join(selected) + ' ' + suffix


def generate_rules(n=1000):
    """Generate n random rules."""
    categories = ['Food', 'Shopping', 'Transport', 'Entertainment', 'Utilities',
                  'Health', 'Travel', 'Subscriptions', 'Home', 'Auto']
    subcategories = ['Grocery', 'Restaurants', 'Online', 'Retail', 'Gas', 'Streaming',
                     'Rideshare', 'Hotels', 'Airlines', 'Services']

    merchants = ['Amazon', 'Uber', 'Netflix', 'Costco', 'Starbucks', 'Target',
                 'Walmart', 'Whole Foods', 'Trader Joes', 'Safeway', 'Shell',
                 'DoorDash', 'Grubhub', 'Apple', 'Google', 'Microsoft']

    rules = []
    for i in range(n):
        merchant = random.choice(merchants)
        pattern = merchant.upper().replace(' ', '')[:6]  # Short pattern

        # Some rules have multiple conditions (more specific)
        if random.random() < 0.3:
            match_expr = f'contains("{pattern}") and amount > {random.randint(10, 100)}'
        else:
            match_expr = f'contains("{pattern}")'

        rule = MerchantRule(
            name=f'{merchant}_{i}',
            match_expr=match_expr,
            category=random.choice(categories),
            subcategory=random.choice(subcategories),
            merchant=merchant,
            tags={'test', f'rule_{i}'},
            line_number=i,
        )
        rules.append(rule)

    return rules


def generate_transactions(n=10000):
    """Generate n random transactions."""
    transactions = []
    for i in range(n):
        txn = {
            'description': generate_random_description(),
            'amount': random.uniform(5, 500),
            'date': date(2025, random.randint(1, 12), random.randint(1, 28)),
            'source': 'TEST',
        }
        transactions.append(txn)
    return transactions


def benchmark():
    """Run performance benchmark."""
    print("Generating test data...")

    rules = generate_rules(1000)
    transactions = generate_transactions(10000)

    print(f"  Rules: {len(rules)}")
    print(f"  Transactions: {len(transactions)}")

    # Create engine and add rules
    engine = MerchantEngine()
    engine.rules = rules

    print("\nBenchmarking match() performance...")

    # Warm up
    for txn in transactions[:100]:
        engine.match(txn)

    # Benchmark
    start = time.perf_counter()
    results = []
    for txn in transactions:
        result = engine.match(txn)
        results.append(result)
    elapsed = time.perf_counter() - start

    # Stats
    matched = sum(1 for r in results if r.matched)
    total_tags = sum(len(r.tags) for r in results)

    print(f"\nResults:")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Per transaction: {elapsed/len(transactions)*1000:.3f}ms")
    print(f"  Transactions/sec: {len(transactions)/elapsed:.0f}")
    print(f"  Matched: {matched}/{len(transactions)} ({matched/len(transactions)*100:.1f}%)")
    print(f"  Total tags assigned: {total_tags}")

    return elapsed


if __name__ == '__main__':
    benchmark()
