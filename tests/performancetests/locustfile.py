from locust import HttpUser, task, between
import random

class FastAPILoadTest(HttpUser):
    # Simulate users waiting between 1 and 3 seconds after each task
    wait_time = between(1, 3)

    @task(1)
    def test_root(self):
        """
        Tests the GET / endpoint.
        Weight is 1, so it runs less frequently than read_item.
        """
        self.client.get("/")

    @task(3)
    def test_read_item(self):
        """
        Tests the GET /items/{item_id} endpoint.
        Weight is 3, so this runs roughly 3x as often as test_root.
        """
        # Generate a random ID to simulate real traffic patterns
        item_id = random.randint(1, 1000)
        
        # dynamic URL: /items/1, /items/45, etc.
        # name="/items/[id]": Groups these in the UI so you get one aggregate stat
        self.client.get(f"/items/{item_id}", name="/items/[id]")