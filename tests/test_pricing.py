import copy
import unittest

import benchmark as b
from test_benchmark import case, config, prediction, profile


class PricingTests(unittest.TestCase):
    def test_blank_current_prices_preserve_saved_rates_and_cached_input_discount(self):
        model = {**profile("gpt",api="openai",endpoint="https://api.openai.com/v1/responses"),
                 "deployment":"hosted", "pricing_per_million":{"input":2,"output":8,"cached_input":.2},
                 "pricing_currency":"USD","pricing_lookup":{"source":"saved source"}}
        archived = config(model)
        current = copy.deepcopy(archived)
        current["models"][0].pop("pricing_per_million")
        original = copy.deepcopy(archived)
        quote,source = b.quotation("gpt",archived,current)
        self.assertIn("saved rates",source)
        self.assertEqual(quote["pricing_lookup"],model["pricing_lookup"])
        self.assertAlmostEqual(b.api_cost({"input_tokens":100,"cached_input_tokens":50,"output_tokens":20},quote),.00027)
        self.assertEqual(archived,original)
        current["models"][0]["pricing_per_million"]={"input":0,"output":0}
        quote,_ = b.quotation("gpt",archived,current)
        self.assertEqual(b.api_cost({},quote),0)

    def test_provider_change_does_not_replace_saved_rates(self):
        model = {**profile("gpt"),"deployment":"hosted","pricing_per_million":{"input":2,"output":8}}
        archived = config(model)
        current = copy.deepcopy(archived)
        current["models"][0].update(endpoint="https://different.example/v1/systemone",pricing_per_million={"input":99,"output":99})
        quote,source = b.quotation("gpt",archived,current)
        self.assertEqual(quote["pricing_per_million"],model["pricing_per_million"])
        self.assertIn("endpoint differs",source)

    def test_jev_published_price_requires_exact_provider_and_returned_version(self):
        model = {**profile("jev",endpoint="https://api.typesafe.ai/v1/systemone"),
                 "model":"jev-latest","deployment":"hosted"}
        row = {"input_tokens":1000,"output_tokens":None,"resolved_model":"jev-1.13.0"}
        self.assertAlmostEqual(b.api_cost(row,model),.000042)
        self.assertIsNone(b.api_cost({**row,"resolved_model":"jev-2.0.0"},model))
        self.assertIsNone(b.api_cost({**row,"resolved_model":None},model))
        self.assertIsNone(b.api_cost({**row,"input_tokens":None},model))
        self.assertIsNone(b.api_cost(row,{**model,"endpoint":"https://gateway.example/v1/systemone"}))
        manual = {**model,"pricing_per_million":{"input":1,"output":0}}
        self.assertAlmostEqual(b.api_cost(row,manual),.001)

    def test_repricing_saved_jev_rows_records_provenance_and_full_cost_coverage(self):
        model = {**profile("jev",endpoint="https://api.typesafe.ai/v1/systemone"),
                 "model":"jev-latest","deployment":"hosted"}
        c = case()
        row = {**prediction(c,alias="jev"),"resolved_model":"jev-1.13.0","input_tokens":1000}
        original = copy.deepcopy(row)
        summaries,priced,_,quotes,*_ = b.summarize([row],[c],config(model))
        self.assertEqual(summaries[0]["cost_coverage"],1)
        self.assertAlmostEqual(summaries[0]["api_per_1k"],.042)
        self.assertEqual(quotes["jev"]["pricing_lookup"]["url"],"https://docs.typesafe.ai/models")
        total = b.cost_totals(priced,quotes)[0]
        self.assertEqual(total["priced_requests"],1)
        self.assertAlmostEqual(total["estimated_api_cost"],.000042)
        html = b.html_report(summaries,{}, {"analysis_costs":[total]})
        self.assertIn("Estimated API cost for this run",html)
        self.assertIn("published Jev 1.13",html)
        self.assertEqual(row,original)


if __name__ == "__main__":
    unittest.main()
