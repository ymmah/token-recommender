#standardSQL
with tokens as (
  select *
  from token_recommender.tokens as tokens
  where true
    and tokens.symbol is not null
    and tokens.price is not null and tokens.price > 0
    and tokens.eth_address is not null
    and tokens.decimals is not null and tokens.decimals >= 0
),
token_balances as (
    with double_entry_book as (
        select token_address, to_address as address, cast(value as float64) as value, block_timestamp
        from `bigquery-public-data.ethereum_blockchain.token_transfers`
        union all
        select token_address, from_address as address, -cast(value as float64) as value, block_timestamp
        from `bigquery-public-data.ethereum_blockchain.token_transfers`
    )
    select double_entry_book.token_address, address, sum(value) as balance
    from double_entry_book
    join tokens on tokens.eth_address = double_entry_book.token_address
    where address != '0x0000000000000000000000000000000000000000'
    group by token_address, address
    having balance > 0
),
token_balances_usd as (
    select
        token_address,
        address,
        balance / pow(10, decimals) * price as balance
    from token_balances
    join tokens on tokens.eth_address = token_balances.token_address
),
filtered_token_balances_usd as (
    select *,
        count(1) over (partition by address) as token_count
    from token_balances_usd
    where balance >= 20
)
select
    token_address,
    address as user_address,
    balance as rating
from filtered_token_balances_usd
where true
    and token_count >= 2
