-- reactivation.sql
-- Identifies dormant high-value customers for a re-engagement campaign.
--
-- Target segment definition:
--   - 3 or more lifetime orders (established buyers, not one-time purchasers)
--   - $300+ lifetime value (meaningful revenue relationship)
--   - Last order was 13–24 months ago (dormant but not permanently lapsed)
--   - Not currently in the suppression list
--   - Non-emailable only (for direct/offline channel targeting)
--     Emailable customers are handled via separate email campaign flow
--
-- Output: contact records for the re-engagement audience, split into
-- a 10,000-customer test cell and holdout group for A/B measurement.
--
-- Rationale: Customers who were active but have drifted represent
-- recoverable revenue. The 13–24 month window captures customers
-- who have lapsed but are not yet permanently churned.


CREATE OR REPLACE TABLE campaign_data.vip_reactivation AS (

  WITH

  -- ── Exclude order types that inflate LTV without true consumer intent ────
  -- Replenishment orders (auto-ship), cancellations, and subscription
  -- orders are excluded to ensure LTV reflects genuine purchase behavior.
  replenishment_orders AS (
    SELECT order_id FROM customer_data.order_payments
    WHERE payment_type = 'REP'
  ),
  cancelled_orders AS (
    SELECT original_order_id AS order_id FROM customer_data.order_adjustments
    WHERE adjustment_type = 'Cancel'
  ),
  subscription_orders AS (
    SELECT order_id FROM customer_data.order_line_items
    WHERE sku IN (SELECT sku FROM product_catalog.subscription_skus)
    -- Alternatively: hard-code known subscription SKU list here
  ),
  excluded_orders AS (
    SELECT order_id FROM replenishment_orders
    UNION SELECT order_id FROM cancelled_orders
    UNION SELECT order_id FROM subscription_orders
  ),

  -- ── Core target: dormant high-value customers ────────────────────────────
  qualified_customers AS (
    SELECT DISTINCT
      enterprise_id,
      email_address,
      SUM(merchandise_amount + shipping_amount) AS lifetime_value
    FROM customer_data.vw_order_header AS o
    WHERE
      -- Exclude internal/bulk order sources
      order_source NOT IN (
        '030', '100', '150', '430', '431', '391', '392',
        '393', '394', '348', '198', '008', '014', '058',
        '060', '074', '596', '600'
      )
      -- Exclude disqualified order types
      AND NOT EXISTS (
        SELECT NULL FROM excluded_orders
        WHERE excluded_orders.order_id = o.order_id
      )
      -- Exclude suppressed customers
      AND NOT EXISTS (
        SELECT NULL FROM customer_data.dm_suppression
        WHERE dm_suppression.enterprise_id = o.enterprise_id
      )
    GROUP BY enterprise_id, email_address
    HAVING
      COUNT(order_id) >= 3          -- established buyer (3+ orders)
      AND lifetime_value > 300       -- meaningful revenue relationship
      AND MAX(order_date) BETWEEN    -- dormant window: 13–24 months ago
        DATEADD(MONTH, -24, CURRENT_DATE())
        AND DATEADD(MONTH, -13, CURRENT_DATE())
  ),

  -- ── Separate emailable vs non-emailable ─────────────────────────────────
  -- Emailable customers will be handled by the email campaign team.
  -- This query targets non-emailable customers for offline channels.
  emailable AS (
    SELECT DISTINCT enterprise_id
    FROM qualified_customers AS qc
    JOIN customer_data.email_safelist AS s
      ON UPPER(qc.email_address) = UPPER(s.email_address)
    WHERE s.file_date = (SELECT MAX(file_date) FROM customer_data.email_safelist)
  ),
  non_emailable AS (
    SELECT DISTINCT enterprise_id FROM qualified_customers AS qc
    WHERE NOT EXISTS (
      SELECT NULL FROM emailable WHERE emailable.enterprise_id = qc.enterprise_id
    )
  ),

  -- ── Pull contact information ─────────────────────────────────────────────
  contact_info AS (
    SELECT * FROM (
      SELECT DISTINCT
        enterprise_id,
        current_client_cust_id,
        ROW_NUMBER() OVER (
          PARTITION BY enterprise_id ORDER BY update_date DESC
        ) AS rank,
        first_name,
        last_name,
        address,
        address_line_2,
        city,
        state_abbrev,
        zip5
      FROM customer_data.vw_customer
      WHERE enterprise_id IN (
        SELECT enterprise_id FROM non_emailable
        ORDER BY RANDOM()
        LIMIT 11211   -- pull slightly more than needed to allow for holdout
      )
    ) ranked
    WHERE rank = 1
  )

  -- ── Final output with test/holdout assignment ────────────────────────────
  -- 10,000 test (T) + remainder as holdout (H)
  -- Holdout group receives no campaign; used to measure incremental lift.
  SELECT
    *,
    ROW_NUMBER() OVER (ORDER BY RANDOM()) AS rand_rank,
    CASE WHEN ROW_NUMBER() OVER (ORDER BY RANDOM()) <= 10000 THEN 'T' ELSE 'H' END AS segment
  FROM contact_info

);


-- ── Export test segment ───────────────────────────────────────────────────
-- Pull the holdout group for measurement tracking (do not mail these customers)
SELECT
  segment,
  enterprise_id,
  current_client_cust_id,
  first_name,
  last_name,
  address,
  address_line_2,
  city,
  state_abbrev,
  zip5
FROM campaign_data.vip_reactivation
WHERE segment = 'T'
ORDER BY rand_rank;
