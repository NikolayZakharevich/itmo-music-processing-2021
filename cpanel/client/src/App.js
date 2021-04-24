import './App.css';
import {useState} from "react";
import ContextMenu from "./components/Menu/ContextMenu";
import Dataset from "./components/Dataset/Dataset";
import {BrowserRouter as Router, Route, Switch} from "react-router-dom";
import Home from "./components/Home/Home";


function App() {
    const [isMenuOpened, setMenuOpened] = useState(false)


    const onMenuClicked = () => {
        setMenuOpened(!isMenuOpened)
    }

    return (
        <Router>
            <Switch>
                <Route exact path="/"><Home/></Route>
                <Route path="/data"><Dataset/></Route>
            </Switch>
            <div className='floating'>
                <ContextMenu isMenuOpened={isMenuOpened} onMenuClicked={onMenuClicked}/>
            </div>
        </Router>
    )
}

export default App;
